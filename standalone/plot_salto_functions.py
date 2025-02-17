import mop_utilities as mop

output_path: str = "/Users/pedrobitencourt/Projects/energy_simulations/figures/"
covo_output_path: str = output_path + "salto_covo_5000.png"
fqeromin_output_path: str = output_path + "salto_fqeromin.png"
voco_output_path: str = output_path + "salto_voco.png"

def salto_covo_config(lake_factor: float) -> dict:
    config: dict = {
        'title': 'Salto Cota Volumen (lake factor: ' + str(lake_factor) + ')',
        'xml': f"""          <fCoVo>
			      <funcion tipo="porRangos">
			        <fueraRango>
			          <funcion tipo="poli">0.0</funcion>
			        </fueraRango>
			        <rangos>(0.0;{lake_factor * 3800.0}),({lake_factor * 3800.0};{lake_factor * 30000.0})</rangos>
			        <funcion tipo="poli">30.0, {lake_factor ** (-1) * 0.0024}, {lake_factor ** (-2) * (-2e-07)}</funcion>
			        <funcion tipo="poli">33.96, {lake_factor **(-1) * 6.67e-04}</funcion>
			      </funcion>
			    </fCoVo>""",
    'x_label': 'volume (hm3)',
    'y_label': 'height (m)',
    'x_range': (0, 5000)
    }
    return config

def salto_fqeromin_config(lake_factor: float) -> dict:
    config: dict = { 
        'title': 'Salto Erogado Minimo',
        'xml': f"""
			    <fQEroMin>
			      <ev tipo="const">
			        <funcion tipo="porRangos">
			          <fueraRango>
			            <funcion tipo="poli">140.0</funcion>
			          </fueraRango>
			          <rangos>(29.0;31.0),(31.0;35.5),(35.5;39.0)</rangos>
                <funcion tipo="poli">{lake_factor * 140.0}</funcion>
                <funcion tipo="poli">{lake_factor * 300.0}</funcion>
                <funcion tipo="poli">{lake_factor * -688400.0}, {lake_factor * 19400.0}</funcion>
			        </funcion>
			      </ev>
			    </fQEroMin>

""",
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)',
        'x_range': (0, 50),
    }
    return config

def salto_voco_config(lake_factor: float) -> dict:
    config: dict = {
        'title': 'Salto Volumen Cota (lake factor: ' + str(lake_factor) + ')',
        'xml': f"""
			    <fVoCo>
			      <funcion tipo="porRangos">
			        <fueraRango>
			          <funcion tipo="poli">0.0</funcion>
			        </fueraRango>
			        <rangos>(0.0;36.5),(36.5;50.0)</rangos>
			        <funcion tipo="poli">{lake_factor * 17571.5}, {lake_factor * -1550.6}, {lake_factor * 32.164}</funcion>
			        <funcion tipo="poli">{lake_factor * -50924.0}, {lake_factor * 1500.0}</funcion>
			      </funcion>
			    </fVoCo>""",
        'x_label': 'height (m)',
        'y_label': 'volume (hm3)',
        'x_range': (0, 50),
    }
    return config
    
    
lake_factor_list: list = [0.05, 0.2, 1, 5]

salto_covo_list: list = [salto_covo_config(lake_factor) for lake_factor in lake_factor_list]
salto_voco_list: list = [salto_voco_config(lake_factor) for lake_factor in lake_factor_list]
salto_fqeromin_list: list = [salto_fqeromin_config(lake_factor) for lake_factor in lake_factor_list]

mop.plot_multiple_functions(salto_covo_list, covo_output_path)
mop.plot_multiple_functions(salto_fqeromin_list, fqeromin_output_path)
mop.plot_multiple_functions(salto_voco_list, voco_output_path)
