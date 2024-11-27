from setuptools import setup, find_packages

setup(
    name="vcsparse",                    # Name of the package
    version="0.1.0",                    # Version
    author="Tong Zhou",                 # Your name
    author_email="zt9465@gmail.com", # Your email
    description="Generate fused and vectorizable kernels for sparse tensor programs on CPUs and GPUs", # Short description
    long_description=open("README.md").read(), # Long description from README file
    long_description_content_type="text/markdown", # Type of the long description
    url="https://github.com/tongzhou80/vcsparse", # URL of your project repository
    packages=find_packages(),           # Automatically find the packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",  # Classifiers help users find your project
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',            # Minimum Python version requirement
    install_requires=['ast-transforms'],
)