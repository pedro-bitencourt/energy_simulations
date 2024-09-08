import xml.dom.minidom
import sys

def format_xml(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        xml_string = file.read()

    # Parse the XML string
    dom = xml.dom.minidom.parseString(xml_string)

    # Pretty-print the XML with custom indentation
    pretty_xml = dom.toprettyxml(indent='    ')

    # Remove empty lines (which minidom sometimes adds)
    pretty_xml_lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(pretty_xml_lines)

    # Write the formatted XML to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(pretty_xml)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.xml output_file.xml")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    format_xml(input_file, output_file)
    print(f"Formatted XML has been written to {output_file}")
