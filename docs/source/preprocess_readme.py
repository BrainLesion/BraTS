import re


def preprocess_readme(input_file: str, output_file: str) -> None:
    """Preprocess a README file to replace GitHub admonitions with Sphinx-style admonitions.

    Args:
        input_file (str): original README file
        output_file (str): processed README file

    """
    with open(input_file, "r") as file:
        content = file.read()

    admonition_types = ["IMPORTANT", "NOTE", "TIP", "WARNING", "CAUTION"]

    for ad_type in admonition_types:
        # Replace > [!ad_type] with Sphinx admonition syntax
        content = re.sub(
            r"> \[!"
            + ad_type
            + "\]\s*\n((?:> .*\n)*)",  # Match the > [!ad_type] and subsequent lines
            lambda m: "```{"
            + ad_type
            + "}\n"
            + m.group(1).replace("> ", "").strip()
            + "\n```",  # Replace with MyST syntax
            content,
        )
    # Write the transformed content to the output file
    with open(output_file, "w") as file:
        file.write(content)


if __name__ == "__main__":
    preprocess_readme("../../README.md", "../README_preprocessed.md")
