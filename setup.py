from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aigen",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Agent Workflow Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aigen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "openai-agents>=0.0.2",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "requests>=2.25.1",
        "gradio>=4.0.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "aigen=aigen.ui.cli:cli_entry_point",
        ],
    },
)