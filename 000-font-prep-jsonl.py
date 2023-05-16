import os
import json
from fontTools.ttLib import TTFont

# get script directory
script_directory = os.path.dirname(os.path.realpath(__file__))

# directory containing your fonts
font_directory = os.path.join(script_directory, "fonts")

# output JSONL file
output_file = os.path.join(script_directory, "output.jsonl")

# list to hold font data
font_data = []

# iterate over each file in the directory and rename to lowercase
for filename in os.listdir(font_directory):
    lower_filename = filename.lower()
    if filename != lower_filename:
        os.rename(os.path.join(font_directory, filename), os.path.join(font_directory, lower_filename))

# iterate over each file in the directory
for filename in os.listdir(font_directory):
    # check if the file is a font file
    if filename.endswith('.ttf') or filename.endswith('.otf'):
        # open the font file
        font = TTFont(os.path.join(font_directory, filename))
        # extract font name and style
        font_name = ""
        font_style = ""
        for name in font['name'].names:
            if name.nameID == 1 and not font_name:  # PostScript name for the font
                font_name = name.toUnicode()
            if name.nameID == 2 and not font_style:  # font subfamily (font style)
                font_style = name.toUnicode()
        # replace spaces with underscores
        font_name = font_name.replace(" ", "_")
        font_style = font_style.replace(" ", "_")
        # if font name or style are not found in metadata, use filename as fallback
        if not font_name:
            font_name = os.path.splitext(filename)[0]  # filename without extension
        if not font_style:
            font_style = "Unknown"
        # create a dict with font data
        font_dict = {'file_name': filename, 'font_name': font_name, 'font_style': font_style}
        # add the dict to our list
        font_data.append(font_dict)

# write the font data to a JSONL file
with open(output_file, 'w') as f:
    for entry in font_data:
        json.dump(entry, f)
        f.write('\n')
