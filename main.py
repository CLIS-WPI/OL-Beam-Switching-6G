#!/usr/bin/env python3
"""
Online Learning-based Adaptive Beam Switching for 6G Networks
Main entry point with CLI arguments for CI/CD pipeline
"""

import os
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import tensorflow as tf

# Set environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import modules
from src.config import *
from src.models import MobilityAwareDQN
from src.core import PredictiveStateHistory, PrioritizedReplayBuffer
from src.environment import *
from src.baselines import MAB_UCB, GreedyBaseline


# =================== METRICS ===================
def compute_service_interruptions(current_snr, prev_snr, threshold=SNR_THRESHOLD):
    """Compute number of service interruptions"""
    if prev_snr is None:
        return 0
    interruptions = 0
    for i in range(len(current_snr)):
        if prev_snr[i] > threshold and current_snr[i] < threshold:
            interruptions += 1
    return interruptions


def compute_stability_score(current_beams, prev_beams, current_snr, prev_snr):
    """Compute stability score based on beam switches and SNR variation"""
    if prev_beams is None or prev_snr is None:
        return 0.0
    switches = current_beams != prev_beams
    switch_rate = np.mean(switches)
    snr_variation = np.mean(np.abs(current_snr - prev_snr))
    return switch_rate + 0.1 * snr_variation


def compute_coverage_ratio(snr_values, threshold=SNR_THRESHOLD):
    """Compute fraction of UEs with SNR above threshold"""
    return np.mean(snr_values > threshold)


def compute_fairness_index(snr_values):
    """Compute Jain's fairness index"""
    snr_values = np.array(snr_values)
    snr_values = snr_values[np.isfinite(snr_values) & (snr_values > -20)]
    if len(snr_values) == 0:
        return 0.0
    numerator = np.sum(snr_values) ** 2
    denominator = len(snr_values) * np.sum(snr_values ** 2)
    return numerator / denominator if denominator > 0 else 0.0


def compute_percentile_snr(snr_values, percentile=10):
    """Compute percentile SNR"""
    snr_values = np.array(snr_values)
    snr_values = snr_values[np.isfinite(snr_values)]
    if len(snr_values) == 0:
        return -30.0
    return np.percentile(snr_values, percentile)


def compute_handover_failures(beam_switches, snr_before, snr_after, threshold=10.0):
    """Compute handover failures (SNR drops significantly after switch)"""
    failures = 0
    for i in range(len(beam_switches)):
        if beam_switches[i] and (snr_before[i] - snr_after[i]) > threshold:
            failures += 1
    return failures


def get_mobility_groups(num_ues):
    """Get UE groups by mobility category"""
    slow = [i for i in range(num_ues) if i % 3 == 0]
    medium = [i for i in range(num_ues) if i % 3 == 1]
    fast = [i for i in range(num_ues) if i % 3 == 2]
    return slow, medium, fast


# =================== REWARD COMPUTATION ===================
def compute_reward(snr_all, prev_snr, beams_all, prev_beams, 
                   service_interruptions, coverage_ratio,
                   profile_name="balanced"):
    """Mode-specific reward computation"""
    profile = REWARD_PROFILES[profile_name]
    
    # SNR reward
    avg_snr = np.mean(snr_all)
    threshold = profile["snr_bonus_threshold"]
    snr_reward = np.clip(avg_snr / 8.0, -4, 6)
    
    # Bonus for exceeding threshold
    if avg_snr > threshold:
        snr_reward += 3.0 * (avg_snr - threshold) / 10.0
    
    # Mode-specific stability penalty
    if prev_beams is not None:
        switches = np.sum(beams_all != prev_beams)
        switch_rate = switches / len(beams_all)
        switch_penalty = -switch_rate * profile["switch_penalty"]
        
        if prev_snr is not None:
            snr_variation = np.mean(np.abs(snr_all - prev_snr))
            variation_penalty = -snr_variation * profile["stability_weight"] * 0.5
        else:
            variation_penalty = 0.0
        
        stability_reward = switch_penalty + variation_penalty
    else:
        stability_reward = 0.0
    
    # Service continuity penalty
    continuity_penalty = -service_interruptions * 3.0
    
    # Coverage bonus
    coverage_bonus = coverage_ratio * 15.0
    if profile_name == "high_coverage":
        coverage_bonus *= 2.0
    
    # Fairness bonus
    fairness = compute_fairness_index(snr_all)
    fairness_bonus = fairness * 4.0
    
    # Mode-specific combination
    total_reward = (
        profile["snr_weight"] * (snr_reward + coverage_bonus + continuity_penalty) +
        profile["stability_weight"] * stability_reward +
        profile["fairness_weight"] * fairness_bonus
    )
    
    return total_reward


# =================== ACTION SELECTION ===================
def get_valid_actions(current_beam, max_beam_change, num_beams):
    """Return valid beam indices based on constraint"""
    if max_beam_change >= num_beams:
        return list(range(num_beams))
    
    valid = []
    for b in range(num_beams):
        # Calculate circular distance
        dist = min(abs(b - current_beam), num_beams - abs(b - current_beam))
        if dist <= max_beam_change:
            valid.append(b)
    return valid


def select_action_with_mode_constraints(q_values, current_beam, epsilon, mode_profile):
    """Select action with mode-specific constraints"""
    max_change = mode_profile["max_beam_change"]
    
    if np.random.rand() < epsilon:
        # Exploration with constraints
        valid_actions = get_valid_actions(current_beam, max_change, NUM_BEAMS)
        return np.random.choice(valid_actions)
    else:
        # Exploitation: best valid action
        valid_actions = get_valid_actions(current_beam, max_change, NUM_BEAMS)
        valid_q_values = q_values[valid_actions]
        best_idx = np.argmax(valid_q_values)
        return valid_actions[best_idx]


# =================== TRAINING ===================
def train_dqn(replay_buffer, q_network, target_network, optimizer, device):
    """Train DQN on a batch from replay buffer"""
    result = replay_buffer.sample(BATCH_SIZE)
    if result is None:
        return 0.0
    
    states, actions, rewards, next_states, dones, indices, weights = result
    
    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.LongTensor(np.array(actions)).to(device)
    rewards_t = torch.FloatTensor(np.array(rewards)).to(device)
    next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
    dones_t = torch.FloatTensor(np.array(dones)).to(device)
    weights_t = torch.FloatTensor(weights).to(device)
    
    batch_size = states_t.shape[0]
    num_ues = states_t.shape[1] if len(states_t.shape) > 2 else 1
    
    if len(states_t.shape) == 3:
        states_flat = states_t.view(batch_size * num_ues, -1)
        next_states_flat = next_states_t.view(batch_size * num_ues, -1)
        actions_flat = actions_t.view(-1)
    else:
        states_flat = states_t
        next_states_flat = next_states_t
        actions_flat = actions_t
    
    current_q = q_network(states_flat)
    current_q_values = current_q.gather(1, actions_flat.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_online = q_network(next_states_flat)
        next_q_target = target_network(next_states_flat)
        best_actions = next_q_online.argmax(1)
        max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
    
    if len(states_t.shape) == 3:
        rewards_expanded = rewards_t.unsqueeze(1).expand(-1, num_ues).reshape(-1)
        dones_expanded = dones_t.unsqueeze(1).expand(-1, num_ues).reshape(-1)
    else:
        rewards_expanded = rewards_t
        dones_expanded = dones_t
    
    target_q = rewards_expanded + GAMMA * max_next_q * (1 - dones_expanded)
    
    td_errors = (current_q_values - target_q).abs().detach().cpu().numpy()
    if len(states_t.shape) == 3:
        td_errors_per_transition = td_errors.reshape(batch_size, num_ues).mean(axis=1)
    else:
        td_errors_per_transition = td_errors
    
    replay_buffer.update_priorities(indices, td_errors_per_transition)
    
    if len(weights_t.shape) == 1 and len(current_q_values.shape) == 1:
        if weights_t.shape[0] != current_q_values.shape[0]:
            weights_t = weights_t.unsqueeze(1).expand(-1, num_ues).reshape(-1)
    
    loss = (weights_t * F.mse_loss(current_q_values, target_q, reduction='none')).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print progress bar"""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


# =================== CLI ARGUMENTS ===================
def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="6G Beam Switching Experiment")
    
    # SYSTEM ARGS
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    
    # REVIEWER 2: Select Profile (Balanced vs Vanilla)
    parser.add_argument("--mode", type=str, default="balanced", 
                        choices=["high_stability", "balanced", "vanilla", "high_coverage"],
                        help="Reward profile to use")
    
    # REVIEWER 1 & 3: Ablation Flag
    parser.add_argument("--ablation", action="store_true", 
                        help="Disable history features to test GRU importance")
    
    # REVIEWER 1 & 2: Latency Flag
    parser.add_argument("--measure_latency", action="store_true", 
                        help="Measure inference time per decision")
    
    return parser.parse_args()


# =================== MAIN ===================
def main():
    args = get_args()
    
    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU: {gpus[0]}")
        except Exception as e:
            print(f"GPU error: {e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {device}")
    
    # Load Profile
    profile = REWARD_PROFILES[args.mode]
    print("="*70)
    print(f"Running Mode: {profile['name']}")
    print(f"Description: {profile['description']}")
    if args.ablation:
        print("WARNING: ABLATION MODE ON (History features disabled)")
    if args.measure_latency:
        print("LATENCY MEASUREMENT ENABLED")
    print("="*70)
    
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for scenario in BLOCKAGE_SCENARIOS:
        print(f"\nScenario: {scenario['name']}")
        P_BB = scenario['P_BB']
        P_UB = scenario['P_UB']
        
        profile_results = defaultdict(lambda: defaultdict(list))
        
        for run in range(args.runs):
            print(f"\n[Run {run+1}/{args.runs}]")
            start_time = time.time()
            
            seed = BASE_SEED + run * 100
            np.random.seed(seed)
            torch.manual_seed(seed)
            tf.random.set_seed(seed)
            random.seed(seed)
            
            q_network = MobilityAwareDQN(STATE_SIZE, hidden_size=HIDDEN_SIZE).to(device)
            target_network = MobilityAwareDQN(STATE_SIZE, hidden_size=HIDDEN_SIZE).to(device)
            target_network.load_state_dict(q_network.state_dict())
            optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
            
            replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
            state_history = PredictiveStateHistory(NUM_UES)
            greedy_baseline = GreedyBaseline()
            mab_ucb = MAB_UCB(NUM_UES, NUM_BEAMS)
            
            # Mode-specific epsilon
            epsilon = profile["epsilon_start"]
            epsilon_min = profile["epsilon_min"]
            epsilon_decay = profile["epsilon_decay"]
            
            prev_beams = None
            prev_snr = np.zeros(NUM_UES)
            prev_blocked = np.zeros(NUM_UES, dtype=bool)
            
            # TRAINING
            print("Training...")
            for t in range(NUM_TIMESTEPS):
                positions, velocities = update_positions(t)
                angles = compute_relative_angles(positions)
                h_channel = generate_fast_channel(positions)
                
                _, additional_blockage_prob = dynamic_channel_conditions(t)
                current_blocked = np.zeros(NUM_UES, dtype=bool)
                for i in range(NUM_UES):
                    adjusted_p_bb = min(1.0, P_BB + additional_blockage_prob)
                    adjusted_p_ub = min(1.0, P_UB + additional_blockage_prob)
                    
                    if prev_blocked[i]:
                        current_blocked[i] = np.random.rand() < adjusted_p_bb
                    else:
                        current_blocked[i] = np.random.rand() < adjusted_p_ub
                
                states = []
                for i in range(NUM_UES):
                    base_features = [
                        angles[i] / np.pi,
                        prev_snr[i] / 40.0,
                        (positions[i] - BS_POSITION) / ROAD_LENGTH,
                    ]
                    # REVIEWER 1 & 3 REQUEST: Pass ablation flag
                    enhanced_features = state_history.get_enhanced_features(i, ablation_mode=args.ablation)
                    state = np.concatenate([base_features, enhanced_features])
                    states.append(state)
                
                states = np.array(states, dtype=np.float32)
                
                # ACTION SELECTION WITH MODE CONSTRAINTS (Batched for GPU efficiency)
                actions = np.zeros(NUM_UES, dtype=int)
                with torch.no_grad():
                    # Batch all UEs together for single GPU forward pass
                    states_tensor = torch.FloatTensor(states).to(device)
                    all_q_values = q_network(states_tensor).cpu().numpy()
                    
                    for i in range(NUM_UES):
                        current_beam = prev_beams[i] if prev_beams is not None else 0
                        actions[i] = select_action_with_mode_constraints(
                            all_q_values[i], current_beam, epsilon, profile
                        )
                
                snr_values = np.zeros(NUM_UES)
                for i in range(NUM_UES):
                    attenuation = BLOCKAGE_ATTENUATION_DB if current_blocked[i] else 0.0
                    snr_values[i] = compute_snr_from_channel(h_channel, actions[i], i, attenuation)
                
                for i in range(NUM_UES):
                    mab_action = mab_ucb.select_beam(i, t)
                    attenuation = BLOCKAGE_ATTENUATION_DB if current_blocked[i] else 0.0
                    mab_snr = compute_snr_from_channel(h_channel, mab_action, i, attenuation)
                    mab_ucb.update(i, mab_action, mab_snr / 10.0)
                
                interruptions = compute_service_interruptions(snr_values, prev_snr)
                coverage = compute_coverage_ratio(snr_values)
                
                reward = compute_reward(
                    snr_values, prev_snr, actions, prev_beams, 
                    interruptions, coverage, args.mode
                )
                
                state_history.update(snr_values, actions, current_blocked, positions, velocities)
                
                next_positions, _ = update_positions(t + 1)
                next_angles = compute_relative_angles(next_positions)
                
                next_states = []
                for i in range(NUM_UES):
                    base_features = [
                        next_angles[i] / np.pi,
                        snr_values[i] / 40.0,
                        (next_positions[i] - BS_POSITION) / ROAD_LENGTH,
                    ]
                    # REVIEWER 1 & 3 REQUEST: Pass ablation flag
                    enhanced_features = state_history.get_enhanced_features(i, ablation_mode=args.ablation)
                    next_state = np.concatenate([base_features, enhanced_features])
                    next_states.append(next_state)
                
                next_states = np.array(next_states, dtype=np.float32)
                
                done = (t == NUM_TIMESTEPS - 1)
                replay_buffer.add(states, actions, reward, next_states, done)
                
                if len(replay_buffer) >= BATCH_SIZE:
                    # Single training step per timestep (efficient)
                    loss = train_dqn(replay_buffer, q_network, target_network, optimizer, device)
                    
                    if (t + 1) % TARGET_UPDATE_FREQ == 0:
                        target_network.load_state_dict(q_network.state_dict())
                
                prev_beams = actions.copy()
                prev_snr = snr_values.copy()
                prev_blocked = current_blocked.copy()
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                
                if (t + 1) % 500 == 0:
                    print_progress_bar(t + 1, NUM_TIMESTEPS, 
                                     prefix=f'  Training', 
                                     suffix=f'ε={epsilon:.3f}')
            
            # EVALUATION
            print("Evaluating...")
            eval_metrics = defaultdict(lambda: defaultdict(list))
            
            prev_beams_drl = None
            prev_beams_greedy = None
            prev_beams_mab = None
            prev_snr_drl = np.zeros(NUM_UES)
            prev_snr_greedy = np.zeros(NUM_UES)
            prev_snr_mab = np.zeros(NUM_UES)
            prev_blocked = np.zeros(NUM_UES, dtype=bool)
            
            slow_ues, medium_ues, fast_ues = get_mobility_groups(NUM_UES)
            
            # REVIEWER 1 & 2 REQUEST: Latency measurement
            inference_times = []
            
            for t in range(EVAL_TIMESTEPS):
                positions, velocities = update_positions(NUM_TIMESTEPS + t)
                angles = compute_relative_angles(positions)
                h_channel = generate_fast_channel(positions)
                
                eval_t = NUM_TIMESTEPS + t
                _, additional_blockage_prob = dynamic_channel_conditions(eval_t)
                current_blocked = np.zeros(NUM_UES, dtype=bool)
                for i in range(NUM_UES):
                    adjusted_p_bb = min(1.0, P_BB + additional_blockage_prob)
                    adjusted_p_ub = min(1.0, P_UB + additional_blockage_prob)
                    
                    if prev_blocked[i]:
                        current_blocked[i] = np.random.rand() < adjusted_p_bb
                    else:
                        current_blocked[i] = np.random.rand() < adjusted_p_ub
                
                states_eval = []
                for i in range(NUM_UES):
                    base_features = [
                        angles[i] / np.pi,
                        prev_snr_drl[i] / 40.0,
                        (positions[i] - BS_POSITION) / ROAD_LENGTH,
                    ]
                    # REVIEWER 1 & 3 REQUEST: Pass ablation flag
                    enhanced_features = state_history.get_enhanced_features(i, ablation_mode=args.ablation)
                    state = np.concatenate([base_features, enhanced_features])
                    states_eval.append(state)
                
                states_eval = np.array(states_eval, dtype=np.float32)
                
                # DRL EVALUATION WITH MODE CONSTRAINTS (Batched for GPU efficiency)
                actions_drl = np.zeros(NUM_UES, dtype=int)
                with torch.no_grad():
                    # Batch all UEs together for single GPU forward pass
                    states_tensor = torch.FloatTensor(states_eval).to(device)
                    
                    # REVIEWER 1 & 2 REQUEST: LATENCY MEASUREMENT
                    if args.measure_latency:
                        start_time_inf = time.perf_counter()
                        all_q_values = q_network(states_tensor)
                        end_time_inf = time.perf_counter()
                        inference_times.append((end_time_inf - start_time_inf) * 1000)  # ms for all UEs
                    else:
                        all_q_values = q_network(states_tensor)
                    
                    all_q_values = all_q_values.cpu().numpy()
                    
                    for i in range(NUM_UES):
                        current_beam = prev_beams_drl[i] if prev_beams_drl is not None else 0
                        # No exploration during evaluation
                        actions_drl[i] = select_action_with_mode_constraints(
                            all_q_values[i], current_beam, 0.0, profile
                        )
                
                actions_greedy = np.array([greedy_baseline.select_beam(h_channel, i) 
                                          for i in range(NUM_UES)])
                
                actions_mab = np.zeros(NUM_UES, dtype=int)
                for i in range(NUM_UES):
                    actions_mab[i] = mab_ucb.select_beam(i, NUM_TIMESTEPS + t)
                
                snr_drl = np.zeros(NUM_UES)
                snr_greedy = np.zeros(NUM_UES)
                snr_mab = np.zeros(NUM_UES)
                
                for i in range(NUM_UES):
                    attenuation = BLOCKAGE_ATTENUATION_DB if current_blocked[i] else 0.0
                    snr_drl[i] = compute_snr_from_channel(h_channel, actions_drl[i], i, attenuation)
                    snr_greedy[i] = compute_snr_from_channel(h_channel, actions_greedy[i], i, attenuation)
                    snr_mab[i] = compute_snr_from_channel(h_channel, actions_mab[i], i, attenuation)
                
                for i in range(NUM_UES):
                    mab_ucb.update(i, actions_mab[i], snr_mab[i] / 10.0)
                
                eval_metrics['drl']['interruptions'].append(
                    compute_service_interruptions(snr_drl, prev_snr_drl))
                eval_metrics['greedy']['interruptions'].append(
                    compute_service_interruptions(snr_greedy, prev_snr_greedy))
                eval_metrics['mab']['interruptions'].append(
                    compute_service_interruptions(snr_mab, prev_snr_mab))
                
                eval_metrics['drl']['coverage'].append(compute_coverage_ratio(snr_drl))
                eval_metrics['greedy']['coverage'].append(compute_coverage_ratio(snr_greedy))
                eval_metrics['mab']['coverage'].append(compute_coverage_ratio(snr_mab))
                
                eval_metrics['drl']['stability'].append(
                    compute_stability_score(actions_drl, prev_beams_drl, snr_drl, prev_snr_drl))
                eval_metrics['greedy']['stability'].append(
                    compute_stability_score(actions_greedy, prev_beams_greedy, snr_greedy, prev_snr_greedy))
                eval_metrics['mab']['stability'].append(
                    compute_stability_score(actions_mab, prev_beams_mab, snr_mab, prev_snr_mab))
                
                eval_metrics['drl']['fairness'].append(compute_fairness_index(snr_drl))
                eval_metrics['greedy']['fairness'].append(compute_fairness_index(snr_greedy))
                eval_metrics['mab']['fairness'].append(compute_fairness_index(snr_mab))
                
                eval_metrics['drl']['avg_snr'].append(np.mean(snr_drl))
                eval_metrics['greedy']['avg_snr'].append(np.mean(snr_greedy))
                eval_metrics['mab']['avg_snr'].append(np.mean(snr_mab))
                
                eval_metrics['drl']['p10'].append(compute_percentile_snr(snr_drl, 10))
                eval_metrics['greedy']['p10'].append(compute_percentile_snr(snr_greedy, 10))
                eval_metrics['mab']['p10'].append(compute_percentile_snr(snr_mab, 10))
                
                if fast_ues:
                    eval_metrics['drl']['fast_coverage'].append(
                        compute_coverage_ratio(snr_drl[fast_ues]))
                    eval_metrics['greedy']['fast_coverage'].append(
                        compute_coverage_ratio(snr_greedy[fast_ues]))
                    eval_metrics['mab']['fast_coverage'].append(
                        compute_coverage_ratio(snr_mab[fast_ues]))
                
                if prev_beams_drl is not None:
                    beam_switches_drl = actions_drl != prev_beams_drl
                    beam_switches_greedy = actions_greedy != prev_beams_greedy
                    beam_switches_mab = actions_mab != prev_beams_mab
                    
                    eval_metrics['drl']['handover_failures'].append(
                        compute_handover_failures(beam_switches_drl, prev_snr_drl, snr_drl))
                    eval_metrics['greedy']['handover_failures'].append(
                        compute_handover_failures(beam_switches_greedy, prev_snr_greedy, snr_greedy))
                    eval_metrics['mab']['handover_failures'].append(
                        compute_handover_failures(beam_switches_mab, prev_snr_mab, snr_mab))
                
                state_history.update(snr_drl, actions_drl, current_blocked, positions, velocities)
                prev_beams_drl = actions_drl.copy()
                prev_beams_greedy = actions_greedy.copy()
                prev_beams_mab = actions_mab.copy()
                prev_snr_drl = snr_drl.copy()
                prev_snr_greedy = snr_greedy.copy()
                prev_snr_mab = snr_mab.copy()
                prev_blocked = current_blocked.copy()
                
                if (t + 1) % 50 == 0:
                    print_progress_bar(t + 1, EVAL_TIMESTEPS, 
                                     prefix='  Evaluation')
            
            # REVIEWER 1 & 2 REQUEST: Print latency results
            if args.measure_latency:
                print(f"\n  Average Inference Latency: {np.mean(inference_times):.4f} ms")
                print(f"  Min Inference Latency: {np.min(inference_times):.4f} ms")
                print(f"  Max Inference Latency: {np.max(inference_times):.4f} ms")
                print(f"  Std Inference Latency: {np.std(inference_times):.4f} ms")
            
            for method in ['drl', 'greedy', 'mab']:
                for metric in eval_metrics[method]:
                    avg_value = np.mean(eval_metrics[method][metric])
                    profile_results[method][metric].append(avg_value)
            
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s")
            print(f"  Stability: {np.mean(eval_metrics['drl']['stability']):.3f}, "
                  f"Coverage: {np.mean(eval_metrics['drl']['coverage']):.1%}")
        
        all_results[scenario['name']][args.mode] = profile_results
    
    # PRINT RESULTS
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for scenario_name in all_results:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"MODE: {profile['name']}")
        print(f"{'='*70}")
        
        metrics_to_show = [
            ('stability', 'Stability Score', 'lower'),
            ('coverage', 'Coverage Ratio', 'higher'),
            ('interruptions', 'Service Interruptions', 'lower'),
            ('fast_coverage', 'Fast UE Coverage', 'higher'),
            ('handover_failures', 'Handover Failures', 'lower'),
            ('avg_snr', 'Average SNR (dB)', 'higher'),
            ('p10', '10th Percentile SNR', 'higher'),
            ('fairness', 'Fairness Index', 'higher'),
        ]
        
        for metric_key, metric_name, direction in metrics_to_show:
            print(f"\n{metric_name}:")
            print("-" * 70)
            
            results = all_results[scenario_name][args.mode]
            
            for method in ['drl', 'greedy', 'mab']:
                if metric_key in results[method]:
                    values = results[method][metric_key]
                    mean = np.mean(values)
                    std = np.std(values)
                    
                    if metric_key in ['coverage', 'fast_coverage']:
                        print(f"  {method.upper():18s}: {mean:6.1%} ± {std:5.1%}")
                    elif metric_key in ['avg_snr', 'p10']:
                        print(f"  {method.upper():18s}: {mean:6.1f} ± {std:4.1f} dB")
                    else:
                        print(f"  {method.upper():18s}: {mean:6.3f} ± {std:5.3f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

