from PIL import Image, ImageDraw, ImageFont
import os
import json

# Specify your own path to the fonts directory
fonts_dir = 'fonts'

# Letters to generate
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Special characters to generate
special_chars = list('!@#$')

# Top most commonly used words of different lengths. You can replace these with your own words
words = {
    3: ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'any', 'all', 'own', 'new', 'her', 'him', 'his', 'get', 'has', 'had', 'was', 'our', 'too', 'two', 'now', 'who', 'why', 'can', 'may', 'say', 'see', 'way', 'day', 'out', 'yes', 'let', 'yet', 'try', 'run', 'old', 'big', 'ask', 'act', 'top', 'hot', 'bad', 'fix', 'job', 'key', 'low', 'map', 'nor', 'odd', 'pay', 'row', 'set', 'use', 'van', 'war', 'zip', 'aid', 'bag', 'cut', 'due', 'ego', 'fan', 'gas', 'hit', 'ink', 'jet', 'kid', 'lab', 'man', 'net', 'off', 'pan', 'quo', 'ram', 'sin', 'tin', 'urn', 'vet', 'wig', 'yak', 'zap', 'arc', 'ban', 'cap', 'dim', 'end', 'fax', 'gem', 'hug', 'ill', 'jog', 'kin', 'log', 'mat', 'nod', 'pot', 'rag', 'sun', 'ten', 'zag'],
    4: ['that', 'with', 'have', 'this', 'will', 'your', 'from', 'when', 'they', 'make', 'time', 'more', 'said', 'them', 'then', 'over', 'well', 'only', 'also', 'want', 'come', 'were', 'look', 'into', 'good', 'back', 'down', 'like', 'just', 'very', 'know', 'need', 'once', 'keep', 'live', 'year', 'work', 'home', 'love', 'take', 'hear', 'give', 'away', 'ever', 'each', 'high', 'mind', 'long', 'best', 'feel', 'help', 'city', 'open', 'side', 'even', 'name', 'next', 'find', 'play', 'most', 'turn', 'read', 'mean', 'last', 'stay', 'left', 'full', 'face', 'move', 'seem', 'such', 'word', 'same', 'line', 'part', 'form', 'much', 'life', 'head', 'hand', 'fact', 'book', 'plan', 'grow', 'lose', 'meet', 'idea', 'end', 'big', 'few', 'lot', 'eye', 'sort', 'near', 'past', 'site', 'case', 'care', 'lead', 'base', 'rise', 'wait', 'drop', 'hair', 'talk', 'land', 'hard', 'mark', 'card', 'blue', 'rest', 'late', 'top', 'post', 'wall', 'race', 'flow', 'bear', 'door', 'edge', 'main', 'spot', 'test', 'shot', 'wide', 'boat', 'pull', 'list', 'bill', 'rule', 'road', 'cell', 'view', 'wear', 'type', 'draw', 'vote', 'tire', 'text']
        }

# Create a directory to save the images
if not os.path.exists('images'):
    os.makedirs('images')

# Load the JSONL file
with open('output.jsonl', 'r') as f:
    font_data = [json.loads(line) for line in f]

for entry in font_data:
    font_file = entry['file_name']
    font_name = entry['font_name']
    font_style = entry['font_style']

    # Load the font
    font = ImageFont.truetype(os.path.join(fonts_dir, font_file), size=200)

    # Generate an image for each letter
    for letter in letters:
        image = Image.new('RGB', (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        w, h = draw.textsize(letter, font=font)
        draw.text(((512-w)/2, (512-h)/2), letter, fill=(0, 0, 0), font=font)
        image_file_name = f'{font_name}-{font_style}-{letter}.png'
        image.save(f'images/{image_file_name}')

        with open(f'images/{image_file_name}.txt', 'w') as f:
            f.write(f"This is '{letter}' in the {font_name} {font_style} font.\n")

    # Generate an image for each special character
    for special_char in special_chars:
        image = Image.new('RGB', (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        w, h = draw.textsize(special_char, font=font)
        draw.text(((512-w)/2, (512-h)/2), special_char, fill=(0, 0, 0), font=font)
        image_file_name = f'{font_name}-{font_style}-{special_char}.png'
        image.save(f'images/{image_file_name}')

        with open(f'images/{image_file_name}.txt', 'w') as f:
            f.write(f"This is '{special_char}' in the {font_name} {font_style} font.\n")

    # Generate an image for each word
    for length, word_list in words.items():
        for word in word_list:
            image = Image.new('RGB', (512, 512), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            w, h = draw.textsize(word, font=font)
            draw.text(((512-w)/2, (512-h)/2), word, fill=(0, 0, 0), font=font)
            image_file_name = f'{font_name}-{font_style}-{word}.png'
            image.save(f'images/{image_file_name}')

            with open(f'images/{image_file_name}.txt', 'w') as f:
                f.write(f"This is '{word}' in the {font_name} {font_style} font.\n")