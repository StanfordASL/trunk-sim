GEOM_COLOR = "0 0 0 1"
PAYLOAD_COLOR = "0 1 0 1"

#TODO: Check if compositeactuated plugin is even necessary
def payload_body(mass=0.1, size=0.1):
    return f'''
        <body name="payload" pos="{size} 0 0">
            <geom name="payload_geom" size="{size}" mass="{mass}" pos="0 0 0" quat="0.707107 0 -0.707107 0" type="sphere" rgba="{PAYLOAD_COLOR}"/>
        </body>
    '''

def body(index, inner="", size=0.05, radius=0.05, density=1000):
    return f'''
        <body name="body_{index}" pos="{size} 0 0">
            <joint name="joint_{index}" pos="0 0 0" type="ball" group="3" actuatorfrclimited="false" armature="0.01" damping="0.25"/>
            <geom name="geom_{index}" size="{radius} {size}" density="{density}" pos="{size} 0 0" quat="0.707107 0 -0.707107 0" type="capsule" rgba="{GEOM_COLOR}"/>
            <!-- Add sides for tendon to pass through-->
            <site name="site_{index}_y_front" pos="0 {radius} 0"/> 
            <site name="site_{index}_y_back" pos="0 -{radius} 0"/>
            <site name="site_{index}_z_front" pos="0 0 {radius}"/> 
            <site name="site_{index}_z_back" pos="0 0 -{radius}"/> 
            <plugin instance="compositeactuated"/>
            {inner}
        </body>
    '''

def tendon(name, site_names, stiffness=1.0, damping=2.0, width=0.002):
    sites = [f'<site site="{site_name}"/>' for site_name in site_names]
    return f'''
        <tendon>
            <spatial name="{name}" stiffness="{stiffness}" damping="{damping}" width="{width}">
                {'\n'.join(sites)}
            </spatial>
        </tendon>
    '''

def muscle(tendon_name):
    return f''' <muscle tendon="{tendon_name}"/> '''

def base(bodies, tendons, muscles):
    return f'''
    <mujoco model="trunk">
        <compiler angle="radian"/>
        <extension>
            <plugin plugin="mujoco.elasticity.cable">
            <instance name="compositeactuated">
                <config key="twist" value="1e8"/>
                <config key="bend" value="1e8"/>
                <config key="vmax" value="0"/>
            </instance>
            </plugin>
        </extension>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1=".6 .8 1" rgb2=".6 .8 1" width="1" height="1"/>
        </asset>

        <worldbody>
            <light pos="0 -1 -1" dir="0 -1 -1" diffuse="1 1 1"/>
            <body name="actuatedB_first" pos="0 0 0.7" quat="0 -0.707107 0 0.707107">
                <!-- Add slider joint with stiffness and damping to simulate elongation of material-->
                <joint name="slider_cable_world" pos="0 0 0" type="slide" axis="1 0 0" group="3" stiffness="1e2" damping="0.8"/>
                <geom name="actuatedG0" size="0.005 0.005" pos="0.005 0 0" quat="0.707107 0 -0.707107 0" type="capsule" rgba="0.8 0.2 0.1 1"/>

                <plugin instance="compositeactuated"/>
                {bodies}
            </body>
        </worldbody>

        {tendons}

        <actuator>
            {muscles}
        </actuator>

    </mujoco>
    '''


def generate_trunk_model(n_bodies = 100, body_size = 10, stiffness = None, armature = 0, damping = 0, num_controls = 0):
    bodies_string = payload_body(mass=0.1, size=0.06)

    for i in range(n_bodies, 0 , -1):
        bodies_string = body(i, inner=bodies_string)

    tendons_string = ""

    muscles_string = ""

    return base(bodies_string, tendons_string, muscles_string)

