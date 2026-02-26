import glob, tqdm, wandb, os, json, random, time, jax, pickle
import imageio.v2 as imageio
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, JsonlLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.rewards.cube_dense_reward import (
    DenseRewardWrapper,
    extract_gripper_pos,
    extract_gripper_gap_from_sim,
    extract_gripper_pad_positions_from_sim,
)

from utils.flax_utils import save_agent, restore_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('tags', 'Default', 'Wandb tag.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('run_name', None, 'Human readable run name for wandb and saving.')
flags.DEFINE_string('project', 'qam-reproduce', 'Wandb project name.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 500000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50000, 'Save interval.') # for the offline stage only.
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/qam.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_num_samples', 0, "Absolute number of dataset samples to use. 0 means use all samples.")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('auto_cleanup', True, "remove all intermediate checkpoints when the run finishes")
flags.DEFINE_bool('save_every_eval', False, 'Save model checkpoint at every evaluation step.')
flags.DEFINE_bool('save_eval_artifacts', False, 'Save eval video + per-frame object pose JSON under save_dir.')

flags.DEFINE_bool('balanced_sampling', False, "sample half offline and online replay buffer")
flags.DEFINE_string(
    'pretrained_actor_path',
    None,
    'Path to pretrained actor checkpoint (e.g., exp/.../params_1000000.pkl). Used to initialize actor from BC-pretrained policy.',
)

flags.DEFINE_string('dense_reward_version', None, 'Dense reward version (v1/v2/v3/v4/v5/v6/v7/v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18/v20/v21/v22/v23/v24/v25/v26/v27/v28/v29/v30), None for original rewards')
flags.DEFINE_float('terminal_bonus', 50.0, 'Terminal success bonus added on success steps for dense rewards (v1-v30).')
flags.DEFINE_float('dense_shaping_lambda', 10.0, 'Shaping coefficient lambda for v4/v5/v6/v7/v8/v10/v11/v12/v13/v14/v15/v16/v17/v18/v20/v21/v22/v23/v24/v25/v26/v27/v28/v29/v30: r=base + lambda*(gamma*Phi(s\')-Phi(s)) + bonus.')
flags.DEFINE_float('v23_step_penalty', 1.0, 'Per-step penalty used by dense reward v23.')
flags.DEFINE_bool(
    'randomize_task_init_cube_pos',
    False,
    'Deprecated/no-op in current setup. Kept for CLI compatibility.',
)
flags.DEFINE_float(
    'cube_success_threshold',
    0.04,
    'Cube success threshold for env success checks and dense reward potential/success.',
)
flags.DEFINE_integer(
    'max_episode_steps',
    0,
    'Override env max episode steps when > 0. 0 keeps the registered default.',
)

def save_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        csv_logger.save(os.path.join(save_dir, f"{prefix}_sv.csv"))

def restore_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        if os.path.exists(os.path.join(save_dir, f"{prefix}_sv.csv")):
            csv_logger.restore(os.path.join(save_dir, f"{prefix}_sv.csv"))


def _save_eval_video_and_pose_artifacts(save_dir, step, renders, render_data):
    """Persist first eval video episode and matching per-frame object poses."""
    if len(renders) == 0:
        return
    frames = renders[0]
    if frames is None or len(frames) == 0:
        return

    video_path = os.path.join(save_dir, f"video_steps_{step}.mp4")
    json_path = os.path.join(save_dir, f"video_steps_{step}.json")

    # Prefer 20fps for stable playback across viewers.
    imageio.mimsave(video_path, frames, fps=20)

    frame_steps = []
    all_frame_steps = render_data.get("frame_steps", [])
    if len(all_frame_steps) > 0:
        fs = all_frame_steps[0]
        try:
            frame_steps = [int(x) for x in fs.tolist()]
        except Exception:
            frame_steps = [int(x) for x in fs]

    object_pose_traces = []
    all_pose_traces = render_data.get("object_pose_traces", [])
    if len(all_pose_traces) > 0:
        object_pose_traces = all_pose_traces[0]

    payload = {
        "step": int(step),
        "video_path": video_path,
        "num_frames": int(len(frames)),
        "frame_steps": frame_steps,
        "frames": object_pose_traces,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved eval artifacts: {video_path} and {json_path}")

class LoggingHelper:
    def __init__(self, csv_loggers, jsonl_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.jsonl_loggers = jsonl_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        assert prefix in self.jsonl_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.jsonl_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS)
    run_name = FLAGS.run_name if FLAGS.run_name is not None else exp_name
    run = setup_wandb(project=FLAGS.project, group=FLAGS.run_group, name=run_name, tags=FLAGS.tags.split(","))
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, run_name)
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        # assert FLAGS.dataset_replace_interval != 0
        # assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        # Detect env_type (single/double/triple/...) for dataset filtering
        _env_type_filter = None
        for _et in ['single', 'double', 'triple', 'quadruple', 'octuple']:
            if f'{_et}-play' in FLAGS.env_name:
                _env_type_filter = _et
                break
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz"))
            if '-val.npz' not in file
            and (('visual' in FLAGS.env_name) == ('visual' in os.path.basename(file)))
            and (_env_type_filter is None or f'-{_env_type_filter}-' in os.path.basename(file))
        ]

        if FLAGS.dataset_proportion < 1.:
            num_datasets = len(dataset_paths)
            num_subset_datasets = max(1, int(num_datasets * FLAGS.dataset_proportion))
            print("actual data proportion:", num_subset_datasets / num_datasets)
            dataset_paths = dataset_paths[:num_subset_datasets]

        env_kwargs = dict()
        if FLAGS.max_episode_steps > 0:
            env_kwargs['max_episode_steps'] = FLAGS.max_episode_steps

        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
            add_info=(FLAGS.dense_reward_version is not None),  # Load qpos/qvel for dense rewards
            **env_kwargs,
        )
    else:
        env_kwargs = dict()
        if FLAGS.max_episode_steps > 0:
            env_kwargs['max_episode_steps'] = FLAGS.max_episode_steps

        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, **env_kwargs)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    
    config = FLAGS.agent
    discount = FLAGS.agent.discount
    config["horizon_length"] = FLAGS.horizon_length

    # Dense reward wrapper
    dense_wrapper = None
    if FLAGS.dense_reward_version is not None:
        print(f"Using dense reward version: {FLAGS.dense_reward_version}")
        dense_wrapper = DenseRewardWrapper(
            task_name=FLAGS.env_name,
            version=FLAGS.dense_reward_version,
            debug=False,
            success_threshold=FLAGS.cube_success_threshold,
            v23_step_penalty=FLAGS.v23_step_penalty,
        )

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - handle dense reward (if enabled)
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )

        if FLAGS.dataset_num_samples > 0:
            capped_size = min(int(FLAGS.dataset_num_samples), len(ds['masks']))
            print(f"Using capped dataset size: {capped_size} / {len(ds['masks'])}")
            ds = Dataset.create(
                **{k: v[:capped_size] for k, v in ds.items()}
            )
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)
        elif dense_wrapper is not None:
            print(f"Computing dense rewards ({FLAGS.dense_reward_version})...")
            if 'qpos' not in ds:
                raise ValueError("Dense rewards require 'qpos' data. Load dataset with add_info=True")
            dense_rewards = dense_wrapper.compute_dataset_rewards(
                ds,
                discount=FLAGS.agent.discount,
                terminal_bonus=FLAGS.terminal_bonus,
                shaping_coef=FLAGS.dense_shaping_lambda,
            )

            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = dense_rewards
            ds = Dataset.create(**ds_dict)
            abs_dense_rewards = np.abs(dense_rewards)
            nonzero_frac = float((abs_dense_rewards > 1e-9).mean())
            dense_stats = {
                "mean": float(dense_rewards.mean()),
                "std": float(dense_rewards.std()),
                "min": float(dense_rewards.min()),
                "max": float(dense_rewards.max()),
                "p01": float(np.quantile(dense_rewards, 0.01)),
                "p50": float(np.quantile(dense_rewards, 0.50)),
                "p99": float(np.quantile(dense_rewards, 0.99)),
                "nonzero_frac": nonzero_frac,
                "mean_abs": float(abs_dense_rewards.mean()),
                "p99_abs": float(np.quantile(abs_dense_rewards, 0.99)),
            }
            print(
                "Dense rewards: "
                f"mean={dense_stats['mean']:.4f}, std={dense_stats['std']:.4f}, "
                f"min={dense_stats['min']:.4f}, max={dense_stats['max']:.4f}, "
                f"p01={dense_stats['p01']:.4f}, "
                f"p50={dense_stats['p50']:.4f}, "
                f"p99={dense_stats['p99']:.4f}, "
                f"nonzero_frac={nonzero_frac:.4f}"
            )
            if FLAGS.dense_reward_version in ("v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                print(
                    "Dense reward delta-mode check: "
                    f"mean_abs={dense_stats['mean_abs']:.6f}, "
                    f"p99_abs={dense_stats['p99_abs']:.6f}"
                )
                print(f"Dense shaping lambda: {FLAGS.dense_shaping_lambda:.4f}")

            if FLAGS.dense_reward_version in ("v26", "v27", "v28", "v29", "v30"):
                targets = dense_wrapper._infer_offline_next_touch_targets(
                    qpos_data=ds_dict['qpos'],
                    obs_data=ds_dict.get('observations', None),
                    terminals_data=ds_dict.get('terminals', None),
                ).astype(np.int32)
                success_counts = np.asarray(
                    [dense_wrapper.count_success_cubes(qpos_i, None) for qpos_i in ds_dict['qpos']],
                    dtype=np.int16,
                )
                ds_dict["cube_targets"] = targets
                ds_dict["success_counts"] = success_counts
                if len(targets) >= FLAGS.horizon_length:
                    max_start = len(targets) - FLAGS.horizon_length
                    probe_n = min(2000, max_start + 1)
                    probe_starts = np.random.randint(0, max_start + 1, size=probe_n)
                    mixed_hits = 0
                    for s in probe_starts:
                        w = targets[s : s + FLAGS.horizon_length]
                        if len(np.unique(w[w >= 0])) >= 2:
                            mixed_hits += 1
                    mixed_rate = float(mixed_hits / probe_n)
                else:
                    mixed_rate = 0.0
                print(
                    f"{FLAGS.dense_reward_version} sampling metadata: "
                    f"targets_known_frac={(targets >= 0).mean():.4f}, "
                    f"mixed_window_frac(h={FLAGS.horizon_length})={mixed_rate:.4f}"
                )
            wandb.log(
                {
                    **{f"dense_reward/{k}": v for k, v in dense_stats.items()},
                    "dense_reward/shaping_lambda": FLAGS.dense_shaping_lambda,
                },
                step=0,
            )

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Load pretrained actor if specified (flow-based algorithms only)
    if FLAGS.pretrained_actor_path is not None:
        print(f"Loading pretrained actor from {FLAGS.pretrained_actor_path}")
        with open(FLAGS.pretrained_actor_path, 'rb') as f:
            pretrained_dict = pickle.load(f)

        pretrained_params = pretrained_dict['agent']['network']['params']
        current_params = agent.network.params
        new_params = dict(current_params)
        loaded = False

        # Find pretrained flow policy (priority order)
        pretrained_flow = None
        for key in ['modules_actor_slow', 'modules_actor_bc_flow', 'modules_actor_flow', 'modules_actor']:
            if key in pretrained_params:
                pretrained_flow = pretrained_params[key]
                print(f"  -> Found pretrained flow policy in {key}")
                break

        if pretrained_flow is None:
            print("Warning: No flow policy found in pretrained checkpoint")
            print(f"  Pretrained keys: {list(pretrained_params.keys())}")
        else:
            # QAM/ATQAM/BAM: actor_slow, actor_fast
            if 'modules_actor_slow' in current_params:
                new_params['modules_actor_slow'] = pretrained_flow
                if 'modules_target_actor_slow' in current_params:
                    new_params['modules_target_actor_slow'] = pretrained_flow
                print("  -> Loaded into actor_slow, target_actor_slow")
                if 'modules_actor_fast' in current_params:
                    new_params['modules_actor_fast'] = pretrained_flow
                    if 'modules_target_actor_fast' in current_params:
                        new_params['modules_target_actor_fast'] = pretrained_flow
                    print("  -> Loaded into actor_fast, target_actor_fast")
                loaded = True

            # FQL/FEdit/DSRL: actor_bc_flow
            if 'modules_actor_bc_flow' in current_params:
                new_params['modules_actor_bc_flow'] = pretrained_flow
                print("  -> Loaded into actor_bc_flow")
                if 'modules_target_actor_bc_flow' in current_params:
                    new_params['modules_target_actor_bc_flow'] = pretrained_flow
                    print("  -> Loaded into target_actor_bc_flow")
                loaded = True

            # FBRAC: actor_flow
            if 'modules_actor_flow' in current_params:
                new_params['modules_actor_flow'] = pretrained_flow
                print("  -> Loaded into actor_flow")
                loaded = True

            # FAWAC/CGQL/IFQL: actor (flow-based)
            flow_actor_agents = ['fawac', 'cgql', 'ifql']
            if 'modules_actor' in current_params and config['agent_name'] in flow_actor_agents:
                new_params['modules_actor'] = pretrained_flow
                print("  -> Loaded into actor")
                if 'modules_target_actor' in current_params:
                    new_params['modules_target_actor'] = pretrained_flow
                    print("  -> Loaded into target_actor")
                loaded = True

        if loaded:
            agent = agent.replace(network=agent.network.replace(params=new_params))
            print("Pretrained actor loaded successfully!")
        else:
            print("Warning: Could not load pretrained actor into current agent.")

    params = agent.network.params
    # filter all target network
    params = {k: v for k, v in params.items() if "target" not in k}

    print(params.keys())
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("param count:", param_count)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    csv_loggers = {prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes}
    jsonl_loggers = {prefix: JsonlLogger(os.path.join(FLAGS.save_dir, f"{prefix}.jsonl"))
                     for prefix in prefixes}

    if os.path.isdir(FLAGS.save_dir):
        print("trying to load from", FLAGS.save_dir)
        if os.path.exists(os.path.join(FLAGS.save_dir, 'token.tk')):
            print("found existing completed run. Exiting...")
            exit()

        try:
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'r') as f:
                progress = f.read()
            
            load_stage, load_step = progress.split(",")
            load_step = int(load_step)
            agent = restore_agent(agent, restore_path=FLAGS.save_dir, restore_epoch=load_step)
            restore_csv_loggers(csv_loggers, FLAGS.save_dir)
            assert load_stage == "offline", "online restoring is not supported"
            success = True
        except:
            success = False
            load_stage = None
            load_step = None
    else:
        success = False
        load_stage = None
        load_step = None

    if not success: # if failed to load, start over
        print("failed to load prev run")
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        flag_dict = get_flag_dict()
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(flag_dict, f)

    logger = LoggingHelper(
        csv_loggers=csv_loggers,
        jsonl_loggers=jsonl_loggers,
        wandb_logger=wandb,
    )

    # Offline RL
    if load_stage == "offline" and load_step is not None:
        start_step = load_step + 1
        print(f"restoring from offline step {start_step}")
    else:
        start_step = 1

    for i in tqdm.tqdm(range(start_step, FLAGS.offline_steps + 1)):
        log_step = i

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
                add_info=(FLAGS.dense_reward_version is not None),  # Load qpos/qvel for dense rewards
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        if config['agent_name'] == 'rebrac':
            agent, offline_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
        else:
            agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        # eval
        if i == FLAGS.offline_steps or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, renders, render_data = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                sparse_reward=FLAGS.sparse,
                dense_wrapper=dense_wrapper,
                dense_discount=FLAGS.agent.discount,
                dense_terminal_bonus=FLAGS.terminal_bonus,
                dense_shaping_lambda=FLAGS.dense_shaping_lambda,
                eval_seed=FLAGS.seed,
            )
            logger.log(eval_info, "eval", step=log_step)
            if len(renders) > 0:
                from log_utils import (
                    get_wandb_video,
                    get_wandb_video_with_reward,
                    get_wandb_video_with_progress,
                    get_wandb_video_with_z_values,
                    get_wandb_video_with_lift_progress,
                    get_wandb_video_with_gripper_cube_dists,
                )
                video = get_wandb_video(renders)
                video_reward = get_wandb_video_with_reward(
                    renders,
                    render_data.get("reward_traces", []),
                    render_data.get("frame_steps", []),
                    chunk_reward_traces=render_data.get("chunk_reward_traces", []),
                )
                payload = {'eval/video': video}
                if video_reward is not None:
                    payload['eval/video_reward'] = video_reward
                if dense_wrapper is not None and dense_wrapper.version in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    video_progress = get_wandb_video_with_progress(
                        renders,
                        render_data.get("progress_traces", []),
                        render_data.get("potential_diff_traces", []),
                        render_data.get("frame_steps", []),
                    )
                    if video_progress is not None:
                        payload['eval/video_progress'] = video_progress
                if dense_wrapper is not None and dense_wrapper.version in ("v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    z_values_video = get_wandb_video_with_z_values(
                        renders,
                        render_data.get("cube_z_traces", []),
                        render_data.get("lower_entry_z_traces", []),
                        render_data.get("frame_steps", []),
                    )
                    if z_values_video is not None:
                        payload['eval/z_values'] = z_values_video
                if dense_wrapper is not None and dense_wrapper.version in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    lift_progress_video = get_wandb_video_with_lift_progress(
                        renders,
                        render_data.get("cube_lift_traces", []),
                        render_data.get("gripper_metric_traces", render_data.get("gripper_width_traces", [])),
                        render_data.get("frame_steps", []),
                        metric_name=render_data.get("gripper_metric_name", "gripper_width"),
                    )
                    if lift_progress_video is not None:
                        payload['eval/lift_progress'] = lift_progress_video
                if dense_wrapper is not None and dense_wrapper.version in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    dist_video = get_wandb_video_with_gripper_cube_dists(
                        renders,
                        render_data.get("left_gripper_cube_dist_traces", []),
                        render_data.get("right_gripper_cube_dist_traces", []),
                        render_data.get("frame_steps", []),
                    )
                    if dist_video is not None:
                        payload["eval/dist"] = dist_video
                logger.wandb_logger.log(payload, step=log_step)
                if FLAGS.save_eval_artifacts:
                    _save_eval_video_and_pose_artifacts(FLAGS.save_dir, log_step, renders, render_data)
            
            print(f"Step {log_step} Evaluation: Success Rate = {eval_info.get('success', 0.0):.4f}")
            if FLAGS.save_every_eval:
                save_agent(agent, FLAGS.save_dir, log_step)
                save_csv_loggers(csv_loggers, FLAGS.save_dir)
            
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)
            save_csv_loggers(csv_loggers, FLAGS.save_dir)
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                f.write(f"offline,{i}")

    # transition from offline to online
    print(train_dataset.keys())
    print(train_dataset["observations"].shape)

    # Strip keys not needed for online training (e.g. qpos, qvel from dense reward datasets)
    # to avoid memory waste and key mismatch in replay buffer add_transition.
    online_keys = {'observations', 'actions', 'rewards', 'terminals', 'masks', 'next_observations'}
    online_dataset = {k: v for k, v in train_dataset.items() if k in online_keys}

    if not FLAGS.balanced_sampling:
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            online_dataset, size=train_dataset.size + FLAGS.online_steps
        )
    else:
        replay_buffer = ReplayBuffer.create(example_batch, size=FLAGS.online_steps)
    
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}
    action_queue = [] # for action chunking
    online_reset_seed_base = int(FLAGS.seed) + 1000000
    online_episode_idx = 0
    try:
        ob, _ = env.reset(seed=online_reset_seed_base + online_episode_idx)
    except TypeError:
        ob, _ = env.reset()
    online_episode_idx += 1

    # Previous state for wrapper-based dense online rewards.
    prev_qpos_dense = env.unwrapped._data.qpos.copy() if dense_wrapper is not None else None
    prev_gripper_gap_dense = extract_gripper_gap_from_sim(env.unwrapped) if dense_wrapper is not None else None
    prev_left_gripper_pos_dense, prev_right_gripper_pos_dense = extract_gripper_pad_positions_from_sim(env.unwrapped) if dense_wrapper is not None else (None, None)
    if dense_wrapper is not None:
        dense_wrapper.set_episode_initial_positions_from_qpos(prev_qpos_dense)
    v8_chunk_shaping = dense_wrapper is not None and dense_wrapper.version in ("v8", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30")
    chunk_start_qpos_dense = None
    chunk_start_ob_dense = None
    chunk_start_gripper_gap_dense = None
    chunk_start_left_gripper_pos_dense = None
    chunk_start_right_gripper_pos_dense = None
    chunk_step_count_dense = 0
    online_success_steps = 0
    online_episode_successes = 0
    online_episode_count = 0
    current_episode_had_success = False
    gamma = FLAGS.agent.discount

    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step = FLAGS.offline_steps + i
        online_rng, key = jax.random.split(online_rng)

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
                add_info=(FLAGS.dense_reward_version is not None),
            )
            train_dataset = process_train_dataset(train_dataset)
            size = train_dataset.size
            
            if FLAGS.balanced_sampling:
                pass
            else:
                for k in train_dataset:
                    replay_buffer[k][:size] = train_dataset[k][:]

        # the action chunk is executed fully
        if len(action_queue) == 0:

            if FLAGS.balanced_sampling and i < FLAGS.start_training:
                action = np.random.rand(action_dim) * 2. - 1.
                action = np.clip(action, -1., 1.)
            else:
                action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
            if v8_chunk_shaping and prev_qpos_dense is not None:
                chunk_start_qpos_dense = prev_qpos_dense.copy()
                chunk_start_ob_dense = np.array(ob, copy=True)
                chunk_start_gripper_gap_dense = prev_gripper_gap_dense
                chunk_start_left_gripper_pos_dense = None if prev_left_gripper_pos_dense is None else prev_left_gripper_pos_dense.copy()
                chunk_start_right_gripper_pos_dense = None if prev_right_gripper_pos_dense is None else prev_right_gripper_pos_dense.copy()
                chunk_step_count_dense = 0
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"): # for cubes
                env_info[key] = value
        step_success = float(bool(info.get("success", False)))
        if step_success > 0:
            current_episode_had_success = True
            online_success_steps += 1
        env_info["success_step"] = step_success
        env_info["success_steps_cum"] = float(online_success_steps)
        env_info["success_step_rate"] = float(online_success_steps / i)
        if dense_wrapper is not None:
            try:
                curr_qpos_for_metrics = env.unwrapped._data.qpos.copy()
                num_success_cubes = dense_wrapper.count_success_cubes(curr_qpos_for_metrics)
                env_info["num_success_cubes"] = float(num_success_cubes)
                env_info["success_cube_fraction"] = float(num_success_cubes / max(dense_wrapper.num_cubes, 1))
                gp_metrics = extract_gripper_pos(next_ob)
                gap_metrics = extract_gripper_gap_from_sim(env.unwrapped)
                left_metrics, right_metrics = extract_gripper_pad_positions_from_sim(env.unwrapped)
                metrics_env = dense_wrapper._make_env()
                metrics_env.load_state(
                    curr_qpos_for_metrics,
                    gripper_pos=gp_metrics,
                    gripper_gap_m=gap_metrics,
                    gripper_gap_open_ref_m=getattr(dense_wrapper, "_v22_gap_open_ref_m", getattr(dense_wrapper, "_v20_gap_open_ref_m", None)),
                    left_gripper_pos=left_metrics,
                    right_gripper_pos=right_metrics,
                )
                metrics_env.use_pad_reach = getattr(dense_wrapper, "version", None) in ("v21", "v22", "v23")
                metrics_env.pad_reach_threshold = 0.07
                active_idx_metrics = dense_wrapper._active_cube_idx(metrics_env)
                if active_idx_metrics is None:
                    env_info["active_cube_idx"] = -1.0
                    env_info["active_cube_lift"] = 0.0
                    env_info["active_cube_lifted"] = 0.0
                    env_info["active_cube_grasped"] = 0.0
                else:
                    cube_metrics = metrics_env.cubes[active_idx_metrics]
                    cube_lift_metrics = float(cube_metrics.position[2] - cube_metrics.initial_position[2])
                    env_info["active_cube_idx"] = float(active_idx_metrics)
                    env_info["active_cube_lift"] = cube_lift_metrics
                    env_info["active_cube_lifted"] = float(cube_lift_metrics > 0.01)
                    env_info["active_cube_grasped"] = float(metrics_env.is_grasped(active_idx_metrics))
            except Exception:
                pass
        if done:
            online_episode_count += 1
            if current_episode_had_success:
                online_episode_successes += 1
            current_episode_had_success = False
        env_info["episode_success_rate"] = float(
            online_episode_successes / max(online_episode_count, 1)
        )
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0
        elif dense_wrapper is not None:
            curr_qpos_dense = env.unwrapped._data.qpos.copy()
            curr_gripper_gap_dense = extract_gripper_gap_from_sim(env.unwrapped)
            curr_left_gripper_pos_dense, curr_right_gripper_pos_dense = extract_gripper_pad_positions_from_sim(env.unwrapped)
            if v8_chunk_shaping:
                base_plus_events = dense_wrapper.compute_online_reward(
                    prev_qpos=prev_qpos_dense,
                    curr_qpos=curr_qpos_dense,
                    env_reward=float(int_reward),
                    prev_ob=ob,
                    curr_ob=next_ob,
                    prev_gripper_gap_m=prev_gripper_gap_dense,
                    curr_gripper_gap_m=curr_gripper_gap_dense,
                    prev_left_gripper_pos=prev_left_gripper_pos_dense,
                    prev_right_gripper_pos=prev_right_gripper_pos_dense,
                    curr_left_gripper_pos=curr_left_gripper_pos_dense,
                    curr_right_gripper_pos=curr_right_gripper_pos_dense,
                    discount=gamma,
                    terminal_bonus=FLAGS.terminal_bonus,
                    shaping_coef=0.0,
                )
                chunk_step_count_dense += 1
                is_chunk_end = (len(action_queue) == 0) or done
                chunk_shaping = 0.0
                if is_chunk_end and chunk_start_qpos_dense is not None and chunk_start_ob_dense is not None:
                    prev_gp = extract_gripper_pos(chunk_start_ob_dense)
                    curr_gp = extract_gripper_pos(next_ob)
                    prev_progress, _ = dense_wrapper.compute_progress(
                        chunk_start_qpos_dense,
                        gripper_pos=prev_gp,
                        gripper_gap_m=chunk_start_gripper_gap_dense,
                        gripper_gap_open_ref_m=getattr(dense_wrapper, "_v22_gap_open_ref_m", getattr(dense_wrapper, "_v20_gap_open_ref_m", None)),
                        left_gripper_pos=chunk_start_left_gripper_pos_dense,
                        right_gripper_pos=chunk_start_right_gripper_pos_dense,
                    )
                    curr_progress, _ = dense_wrapper.compute_progress(
                        curr_qpos_dense,
                        gripper_pos=curr_gp,
                        gripper_gap_m=curr_gripper_gap_dense,
                        gripper_gap_open_ref_m=getattr(dense_wrapper, "_v22_gap_open_ref_m", getattr(dense_wrapper, "_v20_gap_open_ref_m", None)),
                        left_gripper_pos=curr_left_gripper_pos_dense,
                        right_gripper_pos=curr_right_gripper_pos_dense,
                    )
                    chunk_shaping = FLAGS.dense_shaping_lambda * (
                        (gamma ** chunk_step_count_dense) * curr_progress - prev_progress
                    )
                int_reward = float(base_plus_events + chunk_shaping)
                if is_chunk_end:
                    chunk_start_qpos_dense = None
                    chunk_start_ob_dense = None
                    chunk_start_gripper_gap_dense = None
                    chunk_start_left_gripper_pos_dense = None
                    chunk_start_right_gripper_pos_dense = None
                    chunk_step_count_dense = 0
            else:
                int_reward = dense_wrapper.compute_online_reward(
                    prev_qpos=prev_qpos_dense,
                    curr_qpos=curr_qpos_dense,
                    env_reward=float(int_reward),
                    prev_ob=ob,
                    curr_ob=next_ob,
                    prev_gripper_gap_m=prev_gripper_gap_dense,
                    curr_gripper_gap_m=curr_gripper_gap_dense,
                    prev_left_gripper_pos=prev_left_gripper_pos_dense,
                    prev_right_gripper_pos=prev_right_gripper_pos_dense,
                    curr_left_gripper_pos=curr_left_gripper_pos_dense,
                    curr_right_gripper_pos=curr_right_gripper_pos_dense,
                    discount=gamma,
                    terminal_bonus=FLAGS.terminal_bonus,
                    shaping_coef=FLAGS.dense_shaping_lambda,
                )
            prev_qpos_dense = curr_qpos_dense
            prev_gripper_gap_dense = curr_gripper_gap_dense
            prev_left_gripper_pos_dense = None if curr_left_gripper_pos_dense is None else curr_left_gripper_pos_dense.copy()
            prev_right_gripper_pos_dense = None if curr_right_gripper_pos_dense is None else curr_right_gripper_pos_dense.copy()

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            try:
                ob, _ = env.reset(seed=online_reset_seed_base + online_episode_idx)
            except TypeError:
                ob, _ = env.reset()
            online_episode_idx += 1
            action_queue = []  # reset the action queue
            if dense_wrapper is not None:
                prev_qpos_dense = env.unwrapped._data.qpos.copy()
                prev_gripper_gap_dense = extract_gripper_gap_from_sim(env.unwrapped)
                prev_left_gripper_pos_dense, prev_right_gripper_pos_dense = extract_gripper_pad_positions_from_sim(env.unwrapped)
                dense_wrapper.set_episode_initial_positions_from_qpos(prev_qpos_dense)
                chunk_start_qpos_dense = None
                chunk_start_ob_dense = None
                chunk_start_gripper_gap_dense = None
                chunk_start_left_gripper_pos_dense = None
                chunk_start_right_gripper_pos_dense = None
                chunk_step_count_dense = 0
        else:
            ob = next_ob

        min_replay_for_sequence = FLAGS.horizon_length
        if i >= FLAGS.start_training and replay_buffer.size >= min_replay_for_sequence:

            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample_sequence(config['batch_size'] // 2 * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
                replay_batch = replay_buffer.sample_sequence(FLAGS.utd_ratio * config['batch_size'] // 2, 
                    sequence_length=FLAGS.horizon_length, discount=discount)
                
                batch = {k: np.concatenate([
                    dataset_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + dataset_batch[k].shape[1:]), 
                    replay_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + replay_batch[k].shape[1:])], axis=1) for k in dataset_batch}
                
            else:
                batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                            sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            if config['agent_name'] == 'rebrac':
                agent, update_info["online_agent"] = agent.batch_update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, renders, render_data = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                sparse_reward=FLAGS.sparse,
                dense_wrapper=dense_wrapper,
                dense_discount=FLAGS.agent.discount,
                dense_terminal_bonus=FLAGS.terminal_bonus,
                dense_shaping_lambda=FLAGS.dense_shaping_lambda,
                eval_seed=FLAGS.seed,
            )
            logger.log(eval_info, "eval", step=log_step)
            if len(renders) > 0:
                from log_utils import (
                    get_wandb_video,
                    get_wandb_video_with_reward,
                    get_wandb_video_with_progress,
                    get_wandb_video_with_z_values,
                    get_wandb_video_with_lift_progress,
                    get_wandb_video_with_gripper_cube_dists,
                )
                video = get_wandb_video(renders)
                video_reward = get_wandb_video_with_reward(
                    renders,
                    render_data.get("reward_traces", []),
                    render_data.get("frame_steps", []),
                    chunk_reward_traces=render_data.get("chunk_reward_traces", []),
                )
                payload = {'eval/video': video}
                if video_reward is not None:
                    payload['eval/video_reward'] = video_reward
                if dense_wrapper is not None and dense_wrapper.version in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    video_progress = get_wandb_video_with_progress(
                        renders,
                        render_data.get("progress_traces", []),
                        render_data.get("potential_diff_traces", []),
                        render_data.get("frame_steps", []),
                    )
                    if video_progress is not None:
                        payload['eval/video_progress'] = video_progress
                if dense_wrapper is not None and dense_wrapper.version in ("v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    z_values_video = get_wandb_video_with_z_values(
                        renders,
                        render_data.get("cube_z_traces", []),
                        render_data.get("lower_entry_z_traces", []),
                        render_data.get("frame_steps", []),
                    )
                    if z_values_video is not None:
                        payload['eval/z_values'] = z_values_video
                if dense_wrapper is not None and dense_wrapper.version in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    lift_progress_video = get_wandb_video_with_lift_progress(
                        renders,
                        render_data.get("cube_lift_traces", []),
                        render_data.get("gripper_metric_traces", render_data.get("gripper_width_traces", [])),
                        render_data.get("frame_steps", []),
                        metric_name=render_data.get("gripper_metric_name", "gripper_width"),
                    )
                    if lift_progress_video is not None:
                        payload['eval/lift_progress'] = lift_progress_video
                if dense_wrapper is not None and dense_wrapper.version in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
                    dist_video = get_wandb_video_with_gripper_cube_dists(
                        renders,
                        render_data.get("left_gripper_cube_dist_traces", []),
                        render_data.get("right_gripper_cube_dist_traces", []),
                        render_data.get("frame_steps", []),
                    )
                    if dist_video is not None:
                        payload["eval/dist"] = dist_video
                logger.wandb_logger.log(payload, step=log_step)
                if FLAGS.save_eval_artifacts:
                    _save_eval_video_and_pose_artifacts(FLAGS.save_dir, log_step, renders, render_data)
            
            print(f"Step {log_step} Evaluation: Success Rate = {eval_info.get('success', 0.0):.4f}")
            if FLAGS.save_every_eval:
                save_agent(agent, FLAGS.save_dir, log_step)
                save_csv_loggers(csv_loggers, FLAGS.save_dir)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()
    for key, jsonl_logger in logger.jsonl_loggers.items():
        jsonl_logger.close()

    # a token to indicate a successfully finished run
    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    # cleanup
    if FLAGS.auto_cleanup:
        all_files = os.listdir(FLAGS.save_dir)
        for relative_path in all_files:
            full_path = os.path.join(FLAGS.save_dir, relative_path)
            if os.path.isfile(full_path) and relative_path.startswith("params"):
                print(f"removing {full_path}")
                os.remove(full_path)

    wandb.finish()

if __name__ == '__main__':
    app.run(main)
