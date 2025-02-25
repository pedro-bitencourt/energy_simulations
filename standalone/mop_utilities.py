import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

def parse_xml_function(xml_input: str):
    # Check function type
    if "poliConCotas" in xml_input:
        return parse_poly_with_bounds_function(xml_input)
    elif "porRangos" in xml_input:
        return parse_by_ranges_function(xml_input)
    else:
        raise ValueError("Unknown function type in XML input")

def grab_xml_field(xml_input: str, field_name: str) -> str:
    match = re.search(f'<{field_name}>(.*?)</{field_name}>', xml_input)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Field '{field_name}' not found in XML input")

def parse_poly_with_bounds_function(xml_input: str):
    # Parse bounds and coefficients
    xmin = float(grab_xml_field(xml_input, 'xmin'))
    xmax = float(grab_xml_field(xml_input, 'xmax'))
    valmin = float(grab_xml_field(xml_input, 'valmin'))
    valmax = float(grab_xml_field(xml_input, 'valmax'))
    coefs = list(map(float, grab_xml_field(xml_input, 'coefs').split(',')))

    def poly_with_bounds_function(x: float) -> float:
        if x < xmin:
            return valmin
        elif x > xmax:
            return valmax
        else:
            value = sum(coef * x**i for i, coef in coefs)
            return max(valmin, min(valmax, value))
    return poly_with_bounds_function

def parse_by_ranges_function(xml_input: str):
    # Parse ranges
    ranges_match = grab_xml_field(xml_input, 'rangos')
    print(ranges_match)
    print(type(ranges_match))

    if ranges_match:
        ranges = [tuple(map(float, r.strip('()').split(';'))) for r in ranges_match.split(',')]
    else:
        ranges = []
        
    ranges = [tuple(map(float, r.strip('()').split(';'))) for r in ranges_match.split(',')]

    # Parse polynomials
    poly_matches = re.findall(r'<funcion tipo="poli">(.*?)</funcion>', xml_input)
    polynomials = [list(map(float, poly.split(','))) for poly in poly_matches]

    print("Polynomials found:")
    print(polynomials)
    print("Ranges found:")
    print(ranges)

    default_polynomial = polynomials[0]
    # Remove default polynomial from the list
    polynomials = polynomials[1:]
    
    def by_ranges_function(x: float) -> float:
        if ranges:
            for (lower, upper), poly in zip(ranges, polynomials):
                if lower <= x < upper:
                    return polynomial_from_list(x, poly)

        return polynomial_from_list(x, default_polynomial)

    return by_ranges_function

def polynomial_from_list(x: float, poly: list[float]):
    idx = 0
    output = 0
    for coef in poly:
        output += coef * x**idx
        idx += 1
    return output

def plot_multiple_functions(function_data: List[Dict], output_path: str ):
    num_functions = len(function_data)
    fig, axes = plt.subplots(num_functions, 1, figsize=(10, 6 * num_functions), squeeze=False)
    
    for i, data in enumerate(function_data):
        xml_input = data['xml']
        x_label = data.get('x_label', 'x')
        y_label = data['y_label']
        title = data['title']
        
        parsed_function = parse_xml_function(xml_input)
        
        # Determine x_range from the XML if not provided
        if 'x_range' in data:
            x_range = data['x_range']
        else:
            xmin = float(grab_xml_field(xml_input, 'xmin'))
            xmax = float(grab_xml_field(xml_input, 'xmax'))
            x_range = (xmin - 1, xmax + 1)  # Add some padding
        
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = [parsed_function(xi) for xi in x]
        
        ax = axes[i, 0]
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        
    
#    plt.tight_layout()
#    plt.show()
    plt.savefig(output_path)
    print("Plot saved to", output_path)


    
# Example usage
function_data = [
    {
        'title': 'Baygorria Erogado Minimo',
        'xml': """    <fQEroMin>
        <ev tipo="const">
          <funcion tipo="poliConCotas">
            <xmin>54.1</xmin>
            <xmax>55.2</xmax>
            <valmin>0.0</valmin>
            <valmax>10000.0</valmax>
            <coefs>393228.0, -22508.966, 282.2677</coefs>
          </funcion>
        </ev>
      </fQEroMin>
        """,
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)'
    },
        {
        'title': 'Bonete Erogado Minimo',
        'xml': """
                  <fQEroMin>
            <ev tipo="const">
              <funcion tipo="porRangos">
                <fueraRango>
                  <funcion tipo="poli">0.0</funcion>
                </fueraRango>
                <rangos>(80.7;83.0),(83.0;100.0)</rangos>
                <funcion tipo="poli">1986719.0, -50367.53, 319.1707</funcion>
                <funcion tipo="poli">-78500.0, 1000.0</funcion>
              </funcion>
            </ev>
          </fQEroMin>

        """,
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)',
        'x_range': (80, 100)
    }
]




# Example usage
erogado_min = [
    {
        'title': 'Bonete Erogado Minimo',
        'xml': """
                  <fQEroMin>
            <ev tipo="const">
              <funcion tipo="porRangos">
                <fueraRango>
                  <funcion tipo="poli">0.0</funcion>
                </fueraRango>
                <rangos>(80.7;83.0),(83.0;100.0)</rangos>
                <funcion tipo="poli">1986719.0, -50367.53, 319.1707</funcion>
                <funcion tipo="poli">-78500.0, 1000.0</funcion>
              </funcion>
            </ev>
          </fQEroMin>

        """,
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)'
    },
    {
        'title': 'Baygorria Erogado Minimo',
        'xml': """    <fQEroMin>
	<ev tipo="const">
	  <funcion tipo="poliConCotas">
		<xmin>54.1</xmin>
		<xmax>55.2</xmax>
		<valmin>0.0</valmin>
		<valmax>10000.0</valmax>
		<coefs>393228.0, -22508.966, 282.2677</coefs>
	  </funcion>
	</ev>
  </fQEroMin>
                        """,
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)'
    },
    {
        'title': 'Palmar Erogado Minimo',
        'xml': """ <fQEroMin>
            <ev tipo="const">
              <funcion tipo="poliConCotas">
                <xmin>40.0</xmin>
                <xmax>44.0</xmax>
                <valmin>0.0</valmin>
                <valmax>25000.0</valmax>
                <coefs>-8.5463792E7, 6318431.1, -155791.69, 1281.163</coefs>
              </funcion>
            </ev>
          </fQEroMin>
""",
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)'
    },
        {
        'title': 'Salto Erogado Minimo',
        'xml': """          <fQEroMin>
            <ev tipo="const">
              <funcion tipo="porRangos">
                <fueraRango>
                  <funcion tipo="poli">140.0</funcion>
                </fueraRango>
                <rangos>(29.0;31.0),(31.0;35.5),(35.5;39.0)</rangos>
                <funcion tipo="poli">140.0</funcion>
                <funcion tipo="poli">300.0</funcion>
                <funcion tipo="poli">-688400.0, 19400.0</funcion>
              </funcion>
            </ev>
          </fQEroMin>
""",
        'x_label': 'height (m)',
        'y_label': 'min flow (m3/s)'
    },
]

volumen_cota=[
    {
        'title': 'Bonete Volumen Cota',
        'xml': """                   <fCoVo>
            <ev tipo="const">
              <funcion tipo="poli">70.10384, 0.002658, -3.412E-7, 3.3485E-11, -1.705E-15, 3.38E-20</funcion>
            </ev>
          </fCoVo>
          <fVoCo>
            <ev tipo="const">
              <funcion tipo="poli">1.802867E7, -1162910.0, 30003.7, -386.961, 2.49291, -0.0064081</funcion>
            </ev>
          </fVoCo>


""",
        'x_label': 'volume (m3)',
        'y_label': 'height (m)'
    },
    {
        'title': 'Baygorria Volumen Cota',
        'xml': """                    <fCoVo>
	<ev tipo="const">
	  <funcion tipo="poli">51.998, 0.010101, 3.7375E-5, -3.4306E-7, 1.0827E-9, -1.168E-12</funcion>
	</ev>
  </fCoVo>
""",
        'x_label': 'volume (m3)',
        'y_label': 'height (m)'
    },
    {
        'title': 'Palmar Volumen Cota',
        'xml': """           <fCoVo>
            <ev tipo="const">
              <funcion tipo="poli">36.02, 0.00425, -7.65E-7, 1.13E-9, -1.05E-12, 2.84E-16</funcion>
            </ev>
          </fCoVo>
""",
        'x_label': 'volume (m3)',
        'y_label': 'height (m)'
    },
    {
        'title': 'Salto Volumen Cota',
        'xml': """          <fCoVo>
            <ev tipo="const">
              <funcion tipo="porRangos">
                <fueraRango>
                  <funcion tipo="poli">0.0</funcion>
                </fueraRango>
                <rangos>(0.0;3800.0),(3800.0;30000.0)</rangos>
                <funcion tipo="poli">30.0, 0.0024, -2.0E-7</funcion>
                <funcion tipo="poli">33.96, 6.67E-4</funcion>
              </funcion>
            </ev>
          </fCoVo>
""",
        'x_label': 'volume (m3)',
        'y_label': 'height (m)'
    }
]

if __name__ == '__main__':
    plot_multiple_functions(volumen_cota)
    plot_multiple_functions(erogado_min)
