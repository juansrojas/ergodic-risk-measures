import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from tqdm import tqdm
import pathlib
import random
import torch
import os

from src.rl_main import ReinforcementLearning


class RLExperiments:
    def __init__(self):
        self.rl = None
        self.output_folder = self.pathlib_folder('./output')
        self.current_experiment_id = None
        self.n_runs = None

    @staticmethod
    def pathlib_folder(folder):
        pathlib_folder = pathlib.Path(folder)
        pathlib_folder.mkdir(parents=True, exist_ok=True)
        return str(pathlib_folder)

    @staticmethod
    def set_random_seed(seed):
        torch.manual_seed(seed)  # Sets seed for CPU operations
        torch.cuda.manual_seed_all(seed)  # Sets seed for current GPU and all future GPUs
        np.random.seed(seed)  # Sets seed for NumPy
        random.seed(seed)  # Sets seed for Python's random module
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables cudnn benchmark for reproducibility

    @staticmethod
    def get_ci_color(hex_color, opacity=0.5, blend_weight=0.7):
        # get confidence interval band color

        # Remove '#' if present
        hex_color = hex_color.lstrip('#')

        # Convert hex to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Blend with white (255, 255, 255)
        r_new = int(round(r * (1 - blend_weight) + 255 * blend_weight))
        g_new = int(round(g * (1 - blend_weight) + 255 * blend_weight))
        b_new = int(round(b * (1 - blend_weight) + 255 * blend_weight))
        return f'rgba({r_new}, {g_new}, {b_new}, {opacity})'

    def smart_tqdm(self, iterable, loop_level):
        if loop_level == 'runs':
            self.n_runs = len(iterable)
            if self.n_runs > 1:
                return tqdm(iterable)
            else:
                return iterable
        elif loop_level in ['steps', 'episodes']:
            if self.n_runs > 1:
                return iterable
            else:
                return tqdm(iterable)

    def start_experiment(self, agent, env, state_representation=None, track_data=True, use_wandb=False,
                         wandb_entity='< WANDB ENTITY >', wandb_project='< WANDB PROJECT >'):

        self.rl = ReinforcementLearning(experiment=self.current_experiment_id,
                                        agent_class=agent, environment_class=env, state_representation=state_representation,
                                        track_data=track_data, use_wandb=use_wandb, wandb_entity=wandb_entity, wandb_project=wandb_project)

        if self.rl.agent.value_type == 'nn_pytorch' and 'load_weights' in self.rl.agent.value_network_config.keys():
            value_network_folder = self.pathlib_folder(self.rl.agent.value_network_config['load_weights']['folder'])
            self.rl.agent.value_network_config['load_weights']['file_path'] = os.path.join(
                value_network_folder, self.current_experiment_id + '_value_network.pth',
            )

        if self.rl.agent.policy_type == 'nn_pytorch' and 'load_weights' in self.rl.agent.policy_network_config.keys():
            policy_network_folder = self.pathlib_folder(self.rl.agent.policy_network_config['load_weights']['folder'])
            self.rl.agent.policy_network_config['load_weights']['file_path'] = os.path.join(
                policy_network_folder, self.current_experiment_id + '_policy_network.pth',
            )

        self.rl.agent.load_pytorch_networks()

    def end_experiment(self):
        if self.rl.agent.value_type == 'nn_pytorch' and 'save_weights' in self.rl.agent.value_network_config.keys():
            value_network_folder = self.pathlib_folder(self.rl.agent.value_network_config['save_weights']['folder'])
            self.rl.agent.value_network_config['save_weights']['file_path'] = os.path.join(
                value_network_folder, self.current_experiment_id + '_value_network.pth',
            )
            torch.save(
                self.rl.agent.neural_network_value['network'].state_dict(),
                self.rl.agent.value_network_config['save_weights']['file_path'],
            )

        if self.rl.agent.policy_type == 'nn_pytorch' and 'save_weights' in self.rl.agent.policy_network_config.keys():
            policy_network_folder = self.pathlib_folder(self.rl.agent.policy_network_config['save_weights']['folder'])
            self.rl.agent.policy_network_config['save_weights']['file_path'] = os.path.join(
                policy_network_folder, self.current_experiment_id + '_policy_network.pth',
            )
            torch.save(
                    self.rl.agent.neural_network_policy['network'].state_dict(),
                    self.rl.agent.policy_network_config['save_weights']['file_path'],
                )

        self.rl.rl_end()

    def run_experiment_continuing(self, experiment, agent, env, num_runs, max_steps, step_size, discount, epsilon=None,
                                  state_representation=None, tau_change_step=None, tau_change_value=None,
                                  track_data=True, use_wandb=False, wandb_entity='< WANDB ENTITY >',
                                  wandb_project='< WANDB PROJECT >'):

        all_results = pd.DataFrame()

        # set step sizes
        use_step_size = {'value': step_size['value']}
        if 'policy' in step_size.keys():
            use_step_size['policy'] = step_size['policy'] * step_size['value']
        if 'avg_reward' in step_size.keys():
            use_step_size['avg_reward'] = step_size['avg_reward'] * step_size['value']
        if 'var' in step_size.keys():
            use_step_size['var'] = step_size['var'] * step_size['value']

        for run in self.smart_tqdm(range(num_runs), loop_level='runs'):
            self.current_experiment_id = experiment + '_' + str(int(run))

            self.start_experiment(
                agent=agent,
                env=env,
                state_representation=state_representation,
                track_data=track_data,
                use_wandb=use_wandb,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
            )

            # start first episode
            episode = 0
            self.set_random_seed(1000 * run + episode)
            last_state, last_action = self.rl.rl_start(seed=1000 * run + episode, epsilon=epsilon)
            for step_n in self.smart_tqdm(range(max_steps), loop_level='steps'):
                if experiment == 'tau_rpbp' and step_n == tau_change_step:
                    self.rl.agent.var_quantile = tau_change_value
                reward, state, action, terminal = self.rl.rl_step(last_state, last_action, step_size=use_step_size, epsilon=epsilon, discount=discount)
                last_state = state
                last_action = action

                if terminal:
                    # start next episode
                    episode += 1
                    last_state, last_action = self.rl.rl_start(seed=1000 * run + episode, epsilon=epsilon)

            # get experiment data
            results_df = self.rl.get_data()

            # add to all results
            results_df['run'] = run + 1
            all_results = pd.concat([all_results, results_df], ignore_index=True)
            all_results['experiment'] = experiment

            # cleanup run
            self.end_experiment()

        return all_results

    def get_tau_results_figure(self, experiment, df_dict, n_runs, tau_change_step, rolling_average_amount=1000,
                              x_max=100000, confidence_interval=0.95, save_figure=False, show_figure=True):

        for df_name in df_dict.keys():
            df_dict[df_name]['color_percent_ci'] = self.get_ci_color(df_dict[df_name]['color_percent'])

        fig = go.Figure()
        fig.add_vline(x=tau_change_step, line=dict(color="#1E3765", width=2, dash="dash"))

        # for confidence interval
        z_value = stats.norm.ppf(0.5 + confidence_interval / 2)

        for df_name, df_info in df_dict.items():
            df = df_info['df']

            # Create a container for results
            df_runs = []

            for run_i in tqdm(range(n_runs)):
                # Filter and compute rolling mean in one step
                df_run = df[df['run'] == run_i + 1].reset_index(drop=True)
                rolling_percent = (
                    df_run['state']
                    .eq('blueworld')
                    .rolling(rolling_average_amount, min_periods=1)
                    .mean()[rolling_average_amount:]
                    .values
                )
                df_runs.append(rolling_percent)

            df_runs = np.column_stack(df_runs)  # Convert list to NumPy array for efficient operations

            # Compute mean and confidence intervals
            percent_mean = np.mean(df_runs, axis=1)
            percent_std = np.std(df_runs, axis=1, ddof=1)  # Use ddof=1 for sample standard deviation

            percent_ci_upper = percent_mean + (z_value * (percent_std / np.sqrt(n_runs)))
            percent_ci_lower = percent_mean - (z_value * (percent_std / np.sqrt(n_runs)))

            fig.add_trace(go.Scatter(x=list(range(len(percent_mean))), y=percent_ci_upper, mode='lines',
                                     line=dict(color=df_dict[df_name]['color_percent_ci'], width=0.5),
                                     showlegend=False))

            fig.add_trace(go.Scatter(x=list(range(len(percent_mean))), y=percent_ci_lower, mode='lines',
                                     line=dict(color=df_dict[df_name]['color_percent_ci'], width=0.5),
                                     fill='tonexty', fillcolor=df_dict[df_name]['color_percent_ci'], showlegend=False))

            fig.add_trace(go.Scatter(x=list(range(len(percent_mean))), y=percent_mean, mode='lines', name=df_name,
                                     line=dict(color=df_dict[df_name]['color_percent'], width=2)))

        fig.update_xaxes(title='Time Step', range=[rolling_average_amount, x_max], linewidth=3, mirror=False,
                         ticks='outside', showline=True, linecolor='#262626', gridcolor='rgba(243,243,241, 0.75)',
                         gridwidth=1)

        fig.update_yaxes(title='Percent of Time in<br>Blue World State (x100%)', linewidth=3, mirror=False,
                         ticks='outside', showline=True,
                         linecolor='#262626', gridcolor='rgba(243,243,241, 0.75)', gridwidth=1)

        fig.update_layout(template='plotly_white', height=500, width=1800,
                          font=dict(color='#3F3F3F', size=25, family='times'),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False,)

        if save_figure:
            fig.write_image("./output/" + experiment + "_results.png", scale=3)
            fig.write_html("./output/" + experiment + "_results.html")

        if show_figure:
            fig.show()

        return

    @staticmethod
    def get_dist_samples(dist_name, dist_1, dist_2a, dist_2b, epsilon, n_samples=100000, dist_2_ab_prob=0.5):
        rewards = []

        if dist_name == 'dist_1':
            dist_1_prob = 1 - epsilon + epsilon / 2
        elif dist_name == 'dist_2':
            dist_1_prob = epsilon / 2

        for n in range(n_samples):
            dist_choice = np.random.choice(['dist_1', 'dist_2'], p=[dist_1_prob, 1 - dist_1_prob])
            if dist_choice == 'dist_1':
                reward = np.random.normal(loc=dist_1['mean'], scale=dist_1['stdev'])
            else:
                dist_2 = np.random.choice(['dist2a', 'dist2b'], p=[dist_2_ab_prob, 1 - dist_2_ab_prob])
                if dist_2 == 'dist2a':
                    reward = np.random.normal(loc=dist_2a['mean'], scale=dist_2a['stdev'])
                elif dist_2 == 'dist2b':
                    reward = np.random.normal(loc=dist_2b['mean'], scale=dist_2b['stdev'])

            reward = min(0, reward)
            rewards.append(reward)

        return np.array(rewards)

    def get_s_results_figure(self, experiment, df_dict, quantile, epsilon, env, rolling_average_amount=1000, x_max=100000,
                             confidence_interval=0.95, get_data_by_episode=False, show_ci_legend=False,
                             save_figure=False, show_figure=True):
        fig = go.Figure()

        fig.add_vline(x=40000, line=dict(color="#1E3765", width=2, dash="dash"))
        fig.add_vline(x=80000, line=dict(color="#1E3765", width=2, dash="dash"))

        dist_change_params = env.gym_env.dist_change_params

        dist_change_steps = list(dist_change_params.keys())
        delta_x = dist_change_steps[1] - dist_change_steps[0]

        for x_step in dist_change_params.keys():
            for dist_name in ['dist_1', 'dist_2']:
                samples = self.get_dist_samples(
                    dist_name,
                    dist_change_params[x_step]['dist_1'],
                    dist_change_params[x_step]['dist_2a'],
                    dist_change_params[x_step]['dist_2b'],
                    epsilon
                )

                var = np.quantile(samples, quantile)
                cvar = samples[samples <= var].mean()

                if dist_name == 'dist_1':
                    plot_color = '#EC52A8'
                elif dist_name == 'dist_2':
                    plot_color = '#2FD1FF'

                fig.add_trace(go.Scatter(
                    name='Red World State' if dist_name == 'dist_1' else 'Blue World State',
                    mode='lines',
                    x=[x_step, x_step + delta_x],
                    y=[cvar, cvar],
                    line=dict(color=plot_color, width=2, dash='dash'),
                    legendgroup=dist_name,
                    showlegend=(x_step == 1)
                ))

        if get_data_by_episode:
            yaxis_title = 'Episode Reward CVaR'
            xaxis_title = 'Episode'
        else:
            yaxis_title = 'Reward CVaR'
            xaxis_title = 'Time Step'

        # for confidence interval
        z_value = stats.norm.ppf(0.5 + confidence_interval / 2)

        # we will retain the main plot line data so that we can plot it after (all) confidence intervals
        plot_lines = {}

        runs = df_dict[list(df_dict.keys())[0]]['df']['run'].unique()
        for df_name in df_dict.keys():
            df_runs = pd.DataFrame()
            cvar_cols = []
            for run_i in tqdm(range(len(runs))):
                # get dataframe
                df = df_dict[df_name]['df']
                df = df[df['run'] == runs[run_i]]
                df = df.reset_index(drop=True)

                # get rolling cvar and average
                rolling_cvar = []
                rolling_average = []
                counter = []
                for i in range(rolling_average_amount, len(df)):
                    rewards = df.iloc[i - rolling_average_amount:i]
                    VAR = rewards['reward'].quantile(quantile)
                    rolling_cvar.append(rewards[rewards['reward'] <= VAR]['reward'].mean())
                    rolling_average.append(rewards['reward'].mean())
                    counter.append(i)

                df_runs['cvar_run_' + str(run_i)] = rolling_cvar

                cvar_cols.append('cvar_run_' + str(run_i))

            cvar_mean = df_runs[cvar_cols].mean(axis=1)
            cvar_std = df_runs[cvar_cols].std(axis=1)

            cvar_ci_upper = cvar_mean + (z_value * (cvar_std / np.sqrt(len(runs))))
            cvar_ci_lower = cvar_mean - (z_value * (cvar_std / np.sqrt(len(runs))))

            cvar_color = df_dict[df_name]['color_cvar']

            # plot confidence intervals
            fig.add_trace(go.Scatter(x=counter, y=cvar_ci_upper, mode='lines',
                                     line=dict(color=self.get_ci_color(cvar_color), width=0.5),
                                     showlegend=False, legendgroup=df_name + ' cvar'))

            fig.add_trace(go.Scatter(x=counter, y=cvar_ci_lower, mode='lines', showlegend=show_ci_legend, legendgroup=df_name + ' cvar',
                                     name=df_name + ' ' + str(np.around(100 * confidence_interval, 0)).replace('.0','') + '% CI',
                                     line=dict(color=self.get_ci_color(cvar_color), width=0.5),
                                     fill='tonexty', fillcolor=self.get_ci_color(cvar_color)))

            # retain the main plot line data to plot after (all) confidence intervals
            plot_lines[df_name] = {
                'x': counter,
                'y_cvar': cvar_mean,
                'cvar_color': cvar_color,
            }

        # now plot the main line data
        for df_name in plot_lines.keys():
            fig.add_trace(go.Scatter(x=plot_lines[df_name]['x'], y=plot_lines[df_name]['y_cvar'], mode='lines', name=df_name + ': Reward CVaR',
                                    legendgroup=df_name + ' cvar', showlegend=False, line=dict(color=plot_lines[df_name]['cvar_color'], width=3)))

        fig.update_xaxes(title=xaxis_title, range=[rolling_average_amount, x_max], linewidth=3, mirror=False,
                         ticks='outside', showline=True, linecolor='#262626', gridcolor='rgba(243,243,241, 0.75)', gridwidth=1)

        fig.update_yaxes(title=yaxis_title, linewidth=3, mirror=False, ticks='outside', showline=True, range=[-1.7, -0.45],
                         linecolor='#262626', gridcolor='rgba(243,243,241, 0.75)', gridwidth=1)

        fig.update_layout(template='plotly_white', height=500, width=1900, font=dict(color='#3F3F3F', size=25, family='times'),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        if save_figure:
            fig.write_image(os.path.join(self.output_folder, experiment + "_results.png"), scale=3)
            fig.write_html(os.path.join(self.output_folder, experiment + "_results.html"))

        if show_figure:
                fig.show()

        return

    def get_discount_factor_plot(self, experiment, horizon=6000, save_figure=False, show_figure=True):
        plot_colors = [
            '#AB1368',
            '#007FA3',
            '#00A189',
            '#6D247A',
            '#F1C500',
        ]

        fig = go.Figure()

        counter = 0
        for discount in [0.5, 0.75, 0.9, 0.99, 0.999]:
            y = discount ** np.arange(horizon)
            x = np.arange(horizon)

            fig.add_trace(go.Scatter(
                name=f'γ = {discount}',
                mode='lines',
                x=x,
                y=y,
                line=dict(color=plot_colors[counter], width=4, dash='dash'),
            ))

            counter += 1

        fig.update_xaxes(title='Time Step, ℓ', linewidth=3, mirror=False,
                         ticks='outside', showline=True, linecolor='#262626', gridcolor='rgba(243,243,241, 0.75)', gridwidth=1)

        fig.update_yaxes(title='γ<sup>ℓ</sup>', linewidth=3, mirror=False, ticks='outside', showline=True,
                         linecolor='#262626', gridcolor='rgba(243,243,241, 0.75)', gridwidth=1)

        fig.update_layout(template='plotly_white', height=500, width=1900,
                          font=dict(color='#3F3F3F', size=25, family='times'),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        if save_figure:
            fig.write_image(os.path.join(self.output_folder, experiment + "_results.png"), scale=3)
            fig.write_html(os.path.join(self.output_folder, experiment + "_results.html"))

        if show_figure:
                fig.show()

        return
