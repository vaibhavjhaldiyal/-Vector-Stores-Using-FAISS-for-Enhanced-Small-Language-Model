import re

def advanced_clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip if it's very short or junk
        if len(line) < 30:
            continue

        # Skip typical figure/caption noise
        if re.search(r'(Figure\s*\d+|Table\s*\d+|Source:|Code\sListing)', line, re.IGNORECASE):
            continue

        # Skip if mostly symbols or numbers
        if re.match(r'^[\W\d\s]+$', line):
            continue

        # Skip repeating footer/header patterns
        if re.match(r'Chapter\s+\d+|Page\s+\d+', line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# Load the raw extracted text
with open("three_pieces_of_os.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean the text
cleaned_text = advanced_clean_text(raw_text)

# Save the cleaned version
with open("three_pieces_of_os_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("✅ Cleaned text saved as 'three_pieces_of_os_cleaned.txt'")
