from ece4078.plotly_viz import labeled_slider
import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from IPython.display import IFrame
import numpy as np
import copy

class bot2D():
    def __init__(self):
        # Configuration of 2D robot in world frame, x,y = coordinates in world frame, theta=orientation
        self.x = 0
        self.y = 0
        self.theta = 0
        self.states = []
        
    def reset(self):
        first_state = self.states[0]
        self.x, self.y, self.theta = first_state
        del self.states[:]
        self.states = [first_state]
        
    def get_state(self):
        """Return the current bicycle state. The state is in (x,y,theta) format"""
        return (self.x, self.y, self.theta)
    
    def set_state(self,x=0,y=0,theta=0):
        """Sets the model new state"""
        self.x = x
        self.y = y
        self.theta = theta
        self.states.append([x, y, theta])

    def get_axis(self):
        R = np.array(
            [[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]]
        )
        x_axis = R @ np.array([1.0, 0.0])
        y_axis = R @ np.array([0.0, 1.0])

        o_ = np.array([self.x, self.y])

        return o_ + x_axis, o_ + y_axis

def bicycle_viz(ctx, mybot, interval = 50, dt = 0.05, frames_max = 300, frame_step = 1):
    if frames_max % frame_step != 0:
        raise ValueError("frames_max must be a multiple of frame_step")
    num_frames = frames_max / frame_step

    app = ctx["app"]

    xaxis, yaxis = mybot.get_axis()
    base_fig = go.Figure()
    base_fig.add_trace(
        go.Scatter(x=[mybot.x], y=[mybot.y], mode="lines", line=dict(width=2, color="blue"))
    )
    # Robot axes (trace 1 & 2)
    base_fig.add_trace(
        go.Scatter(x=[mybot.x, xaxis[0]], y=[mybot.y, xaxis[1]], mode="lines", line=dict(width=4, color="red"), showlegend=False)
    )
    base_fig.add_trace(
        go.Scatter(x=[mybot.x, yaxis[0]], y=[mybot.y, yaxis[1]], mode="lines", line=dict(width=4, color="green"), showlegend=False)
    )

    base_fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(range=[-10, 10], scaleanchor="y", scaleratio=1, title="x"),
        yaxis=dict(range=[-10, 10], title="y"),
        showlegend=False,
    )

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "column", "height": "100vh"},
        children=[
            dcc.Graph(id="bicycle-graph", figure=base_fig, style={"flex": "1 1 auto"}),
            dcc.Interval(id="timer", interval=interval, n_intervals=0, disabled=True),
            html.Div(
                style={"display": "flex", "gap": "2rem", "flexWrap": "wrap", "padding": "1rem 10"},
                children=[
                    html.Div(
                        labeled_slider("velocity", "Velocity (v)", -mybot.vel_max, mybot.vel_max, 0.1, 0.0,
                                       marks={-mybot.vel_max: str(-mybot.vel_max), 0: "0", mybot.vel_max: str(mybot.vel_max)}),
                        style={"flex": "1 1 50px", "minWidth": "50px"},
                    ),
                    html.Div(
                        labeled_slider(
                            "delta",
                            "Angle (delta)",
                            -np.degrees(mybot.delta_max),
                            np.degrees(mybot.delta_max),
                            1,
                            0.0,
                            marks={
                                int(-np.degrees(mybot.delta_max)): f"{int(-np.degrees(mybot.delta_max))}°",
                                0: "0°",
                                int(np.degrees(mybot.delta_max)): f"{int(np.degrees(mybot.delta_max))}°",
                            },
                            ),
                        style={"flex": "1 1 50px", "minWidth": "50px"},
                    ),
                    html.Div(id="frame-label", style={"alignSelf": "center", "fontFamily": "monospace"}),
                    html.Button("Play", id="play-btn", n_clicks=0),
                    html.Button("Pause", id="pause-btn", n_clicks=0),
                    html.Button("Reset", id="reset-btn", n_clicks=0),
                ],
            ),
        ],
    )

    @app.callback(
        Output("timer", "disabled"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        State("timer", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_timer(play_clicks, pause_clicks, disabled):
        trig_id = dash.callback_context.triggered_id
        return trig_id != "play-btn"

    @app.callback(
        Output("velocity", "value"),
        Output("delta", "value"),
        Output("timer", "disabled", allow_duplicate=True),
        Output("timer",  "n_intervals"),
        Output("bicycle-graph", "figure", allow_duplicate=True),
        Output("frame-label", "children", allow_duplicate=True),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_state(_):
        mybot.reset()
        fresh_fig = ctx["app"].layout.children[0].figure
        fresh_fig.data[0].x, fresh_fig.data[0].y = [mybot.x], [mybot.y]
        fresh_fig.data[1].x, fresh_fig.data[1].y = [mybot.x, mybot.x + 1], [mybot.y, mybot.y]
        fresh_fig.data[2].x, fresh_fig.data[2].y = [mybot.x, mybot.x], [mybot.y, mybot.y + 1]
        return 0.0, 0.0, True, 0, fresh_fig, f"Frame: 0/{frames_max}"
        

    @app.callback(
        Output("bicycle-graph", "extendData"),
        Output("timer", "disabled", allow_duplicate=True),
        Input("timer", "n_intervals"),
        State("bicycle-graph", "figure"),
        State("velocity", "value"),
        State("delta", "value"),
        prevent_initial_call=True,
    )
    
    def step(frame_idx, fig, v, delta_deg):
        if (frame_idx * frame_step) >= frames_max:
            return no_update, True

        mybot.update_control(v, np.radians(delta_deg))
        x_traj = []
        y_traj = []    
        for _ in range(frame_step):
            mybot.drive(dt=dt)
            x_traj.append(mybot.x)
            y_traj.append(mybot.y)

        xaxis, yaxis = mybot.get_axis()

        patch = {
            "x": [
                x_traj,
                [mybot.x, xaxis[0]],
                [mybot.x, yaxis[0]],
            ],
            "y": [
                y_traj,
                [mybot.y, xaxis[1]],
                [mybot.y, yaxis[1]],
            ],
        }
        return (patch, [0, 1, 2], {"x": [num_frames, 2, 2], "y": [num_frames, 2, 2]}), no_update

    @app.callback(Output("frame-label", "children"), Input("timer", "n_intervals"))
    def update_label(n):
        return f"Frame: {min(n * frame_step, frames_max)}/{frames_max}"


def penguinpi_viz(ctx, mybot, interval = 50, dt = 0.05, frames_max = 300, frame_step = 1):
    if frames_max % frame_step != 0:
        raise ValueError("frames_max must be a multiple of frame_step")
    num_frames = frames_max / frame_step

    app = ctx["app"]

    xaxis, yaxis = mybot.get_axis()
    base_fig = go.Figure()
    base_fig.add_trace(
        go.Scatter(x=[mybot.x], y=[mybot.y], mode="lines", line=dict(width=2, color="blue"))
    )
    # Robot axes (trace 1 & 2)
    base_fig.add_trace(
        go.Scatter(x=[mybot.x, xaxis[0]], y=[mybot.y, xaxis[1]], mode="lines", line=dict(width=4, color="red"), showlegend=False)
    )
    base_fig.add_trace(
        go.Scatter(x=[mybot.x, yaxis[0]], y=[mybot.y, yaxis[1]], mode="lines", line=dict(width=4, color="green"), showlegend=False)
    )

    base_fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(range=[-10, 10], scaleanchor="y", scaleratio=1, title="x"),
        yaxis=dict(range=[-10, 10], title="y"),
        showlegend=False,
    )

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "column", "height": "100vh"},
        children=[
            dcc.Graph(id="bicycle-graph", figure=base_fig, style={"flex": "1 1 auto"}),
            dcc.Interval(id="timer", interval=interval, n_intervals=0, disabled=True),
            html.Div(
                style={"display": "flex", "gap": "2rem", "flexWrap": "wrap", "padding": "1rem 10"},
                children=[
                    html.Div(
                        labeled_slider("velocity", "v", -mybot.max_linear_velocity, mybot.max_linear_velocity, 0.1, 0.0,
                                       marks={-mybot.max_linear_velocity: str(-mybot.max_linear_velocity), 0: "0", mybot.max_linear_velocity: str(mybot.max_linear_velocity)}),
                        style={"flex": "1 1 50px", "minWidth": "50px"},
                    ),
                    html.Div(
                        labeled_slider(
                            "delta",
                            "delta",
                            -np.degrees(mybot.max_angular_velocity),
                            np.degrees(mybot.max_angular_velocity),
                            1,
                            0.0,
                            marks={
                                int(-np.degrees(mybot.max_angular_velocity)): f"{int(-np.degrees(mybot.max_angular_velocity))}°",
                                0: "0°",
                                int(np.degrees(mybot.max_angular_velocity)): f"{int(np.degrees(mybot.max_angular_velocity))}°",
                            },
                            ),
                        style={"flex": "1 1 50px", "minWidth": "50px"},
                    ),
                    html.Div(id="frame-label", style={"alignSelf": "center", "fontFamily": "monospace"}),
                    html.Button("Play", id="play-btn", n_clicks=0),
                    html.Button("Pause", id="pause-btn", n_clicks=0),
                    html.Button("Reset", id="reset-btn", n_clicks=0),
                ],
            ),
        ],
    )

    @app.callback(
        Output("timer", "disabled"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        State("timer", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_timer(play_clicks, pause_clicks, disabled):
        trig_id = dash.callback_context.triggered_id
        return trig_id != "play-btn"

    @app.callback(
        Output("velocity", "value"),
        Output("delta", "value"),
        Output("timer", "disabled", allow_duplicate=True),
        Output("timer",  "n_intervals"),
        Output("bicycle-graph", "figure", allow_duplicate=True),
        Output("frame-label", "children", allow_duplicate=True),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_state(_):
        mybot.reset()
        fresh_fig = ctx["app"].layout.children[0].figure
        fresh_fig.data[0].x, fresh_fig.data[0].y = [mybot.x], [mybot.y]
        fresh_fig.data[1].x, fresh_fig.data[1].y = [mybot.x, mybot.x + 1], [mybot.y, mybot.y]
        fresh_fig.data[2].x, fresh_fig.data[2].y = [mybot.x, mybot.x], [mybot.y, mybot.y + 1]
        return 0.0, 0.0, True, 0, fresh_fig, f"Frame: 0/{frames_max}"
        

    @app.callback(
        Output("bicycle-graph", "extendData"),
        Output("timer", "disabled", allow_duplicate=True),
        Input("timer", "n_intervals"),
        State("bicycle-graph", "figure"),
        State("velocity", "value"),
        State("delta", "value"),
        prevent_initial_call=True,
    )
    
    def step(frame_idx, fig, v, delta_deg):
        if (frame_idx * frame_step) >= frames_max:
            return no_update, True

        mybot.update_control(v, np.radians(delta_deg))
        x_traj = []
        y_traj = []    
        for _ in range(frame_step):
            mybot.drive(dt=dt)
            x_traj.append(mybot.x)
            y_traj.append(mybot.y)

        xaxis, yaxis = mybot.get_axis()

        patch = {
            "x": [
                x_traj,
                [mybot.x, xaxis[0]],
                [mybot.x, yaxis[0]],
            ],
            "y": [
                y_traj,
                [mybot.y, xaxis[1]],
                [mybot.y, yaxis[1]],
            ],
        }
        return (patch, [0, 1, 2], {"x": [num_frames, 2, 2], "y": [num_frames, 2, 2]}), no_update

    @app.callback(Output("frame-label", "children"), Input("timer", "n_intervals"))
    def update_label(n):
        return f"Frame: {min(n * frame_step, frames_max)}/{frames_max}"

def display_bicycle_wheels(rear_wheel, front_wheel, theta):               
    # Initialize figure
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim([0,4])
    ax.set_ylim([0,4])
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.title('Overhead View')
    plt.xlabel('X (m)',weight='bold')
    plt.ylabel('Y (m)',weight='bold')

    ax.plot(0,0)
  
    rear_wheel_x = FancyArrowPatch((0,0), (0.4,0),
                                        mutation_scale=8,color='red')
    rear_wheel_y = FancyArrowPatch((0,0), (0,0.4),
                                        mutation_scale=8,color='red')

    front_wheel_x = FancyArrowPatch((0,0), (0.4,0),
                                        mutation_scale=8,color='blue') 
    front_wheel_y = FancyArrowPatch((0,0), (0,0.4),
                                        mutation_scale=8,color='blue')

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]
    
    # Apply translation and rotation as specified by current robot state
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    Tw_rear = np.eye(3)
    Tw_rear[0:2,2] = rear_wheel
    Tw_rear[0:2,0:2] = [[cos_theta,-sin_theta],[sin_theta,cos_theta]]
    Tw_rear_obj = transforms.Affine2D(Tw_rear)

    Tw_front = np.eye(3)
    Tw_front[0:2,2] = front_wheel
    Tw_front[0:2,0:2] = [[cos_theta,-sin_theta],[sin_theta,cos_theta]]
    Tw_front_obj = transforms.Affine2D(Tw_front)

    ax_trans = ax.transData
    
    rear_wheel_x.set_transform(Tw_rear_obj+ax_trans)
    rear_wheel_y.set_transform(rear_wheel_x.get_transform())
    ax.add_patch(rear_wheel_x)
    ax.add_patch(rear_wheel_y)

    front_wheel_x.set_transform(Tw_front_obj+ax_trans)
    front_wheel_y.set_transform(front_wheel_x.get_transform())
    ax.add_patch(front_wheel_x)
    ax.add_patch(front_wheel_y)

    ax.legend(custom_lines, ['Rear Wheel', 'Front Wheel']) 
