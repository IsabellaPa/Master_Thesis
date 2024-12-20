import re

file = #
clean_file = #

import re

with open(file, "r", encoding="utf-8") as bib_file:
    # Read the entire file into a string
    bib_data = bib_file.read()

    # Replace all occurrences of "abstract = {...}," with an empty string
    bib_data = re.sub(r"abstract\s*=\s*\{[^}]*\},", "", bib_data)

    # Replace all occurrences of "&" with "/&"
    bib_data = re.sub(r"&(?=[^#])", r"\&", bib_data)

with open(clean_file, "w", encoding="utf-8") as cleaned_file:
    # Write the modified string to a new file
    cleaned_file.write(bib_data)

print('done')