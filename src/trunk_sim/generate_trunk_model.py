GEOM_COLOR = "0 0 0 1"
PAYLOAD_COLOR = "0 1 0 1"

def payload_body(mass=0.1, size=0.1):
    return f'''
        <body name="payload" pos="{size} 0 0">
            <geom name="payload_geom" size="{size}" mass="{mass}" pos="0 0 0" quat="0.707107 0 -0.707107 0" type="sphere" rgba="{PAYLOAD_COLOR}"/>
        </body>
    '''

def link(index, inner="", size=0.1, radius=0.005, density=0.1, damping=10.0, armature=0.01, stiffness=100.0, joint=True):
    assert size > radius, "Size must be greater than radius"
    joint_str = f'<joint name="joint_{index}" pos="0 0 0" type="ball" group="3" actuatorfrclimited="false" armature="{armature}" damping="{damping}" stiffness="{stiffness}"/>' if joint else ""
    return f'''
        <body name="link_{index}" pos="{size} 0 0">
            {joint_str}
            <geom name="geom_{index}" size="{radius} {size/2}" density="{density}" pos="{radius} 0 0" quat="0.707107 0 -0.707107 0" type="capsule" rgba="{GEOM_COLOR}"/>
            <site name="site_y_front_{index}" pos="0 {radius} 0"/> 
            <site name="site_y_back_{index}" pos="0 -{radius} 0"/>
            <site name="site_z_front_{index}" pos="0 0 {radius}"/> 
            <site name="site_z_back_{index}" pos="0 0 -{radius}"/>
            {inner}
        </body>
    '''

def tendon(name, site_names, stiffness=0.0, damping=1.0, width=0.002):
    sites = [f'<site site="{site_name}"/>' for site_name in site_names]
    return f'''
        <tendon>
            <spatial name="{name}" stiffness="{stiffness}" damping="{damping}" width="{width}">
                {'\n'.join(sites)}
            </spatial>
        </tendon>
    '''

def muscle(tendon_name):
    return f'''<muscle tendon="{tendon_name}"/> '''

def contact(body_1, body_2):
    return f'''
        <exclude body1="{body_1}" body2="{body_2}"/>
    '''
def base(bodies, tendons="", muscles="", contacts=""):
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

        <contact>
            {contacts}
        </contact>

    </mujoco>
    '''

def generate_trunk_model(num_links=10, tip_mass=0.1, radius=0.025, length=0.32):
    bodies_string = payload_body(mass=tip_mass) if tip_mass > 0 else ""

    size = length/num_links
    for i in range(num_links, 0, -1):
        bodies_string = link(i, inner=bodies_string, size=size, radius=radius)

    bodies_string = link(0, inner=bodies_string, joint=False, size=size, radius=radius)

    tendons_string = ""
    muscles_string = ""
    for position in ["y_front", "y_back", "z_front", "z_back"]:
        tendons_string += tendon(f"tendon_{position}", [f"site_{position}_{i}" for i in range(0, num_links + 1)])

        muscles_string += muscle(f"tendon_{position}")

    contacts = "\n".join([contact(f"link_{i}", f"link_{i+1}") for i in range(0, num_links)])

    return base(bodies_string, tendons=tendons_string, muscles=muscles_string, contacts=contacts)
