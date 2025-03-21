from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="localllmhub",
    version="0.3.1",
    author="Ashraff Hathibelagal",
    description="A powerful, Transformer-based text-to-speech (TTS) tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hathibelagal-dev/localllmhub",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "transformers",
        "requests",
        "torch",
        "uvicorn",
        "asgiref",
        "fastapi",
        "jinja2"
    ],
    entry_points={
        "console_scripts": [
            "localllmhub=localllmhub.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai llm llama gemma qwen local",
    project_urls={
        "Source": "https://github.com/hathibelagal-dev/localllmhub",
        "Tracker": "https://github.com/hathibelagal-dev/localllmhub/issues",
    },
    package_data={
        "localllmhub": ["templates/*.html"]
    },
    include_package_data=True
)
