GEOM_COLOR = "0 0 0 1"
PAYLOAD_COLOR = "0 1 0 1"

# TODO: Check if compositeactuated plugin is even necessary
def payload_body(mass=0.1, size=0.1):
    return f'''
        <body name="payload" pos="{size} 0 0">
            <geom name="payload_geom" size="{size}" mass="{mass}" pos="0 0 0" quat="0.707107 0 -0.707107 0" type="sphere" rgba="{PAYLOAD_COLOR}"/>
        </body>
    '''

def link(index, inner="", size=0.01, radius=0.005, density=0.1, damping=0.25, armature=0.01):
    assert size > radius, "Size must be greater than radius"
    return f'''
        <body name="link_{index}" pos="{size} 0 0">
            <joint name="joint_{index}" pos="0 0 0" type="ball" group="3" actuatorfrclimited="false" armature="{armature}" damping="{damping}"/>
            <geom name="geom_{index}" size="{radius} {radius}" density="{density}" pos="{radius} 0 0" quat="0.707107 0 -0.707107 0" type="capsule" rgba="{GEOM_COLOR}"/>
            <!--<site name="site_{index}_y_front" pos="0 {radius} 0"/> 
            <site name="site_{index}_y_back" pos="0 -{radius} 0"/>
            <site name="site_{index}_z_front" pos="0 0 {radius}"/> 
            <site name="site_{index}_z_back" pos="0 0 -{radius}"/>-->
            {inner}
        </body>
    '''

def tendon(name, site_names, stiffness=0.0, damping=0.0, width=0.002):
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

def base(bodies, tendons="", muscles=""):
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
                <geom name="actuatedG0" size="0.005 0.005" pos="0.005 0 0" quat="0.707107 0 -0.707107 0" type="capsule" rgba="0.8 0.2 0.1 1"/>
                {bodies}
            </body>
        </worldbody>

        {tendons}

        <actuator>
            {muscles}
        </actuator>

    </mujoco>
    '''

def generate_trunk_model(n_links=100, payload_mass=0.1):
    bodies_string = payload_body(mass=payload_mass) if payload_mass > 0 else ""

    for i in range(n_links, 0, -1):
        bodies_string = link(i, inner=bodies_string)

    tendons_string = tendon("tendon_y_front", [f"site_{i}_y_front" for i in range(1, n_links + 1)])
    muscles_string = muscle("tendon_y_front")

    return base(bodies_string)


if __name__ == "__main__":
    model_xml = generate_trunk_model(n_links=10, payload_mass=0.0)
    with open("trunk.xml", "w") as f:
        f.write(model_xml)
