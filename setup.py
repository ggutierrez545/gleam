import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name="glearn",
    version="0.1",
    author="Guillermo Javier Gutierrez",
    author_email="guillo.j.gutierrez@gmail.com",
    description="An explanatory machine learning package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ggutierrez545/glearn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
