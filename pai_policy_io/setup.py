from setuptools import find_packages, setup

package_name = "pai_policy_io"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="Franco Cipollone",
    maintainer_email="franco.cipollone@ekumenlabs.com",
    description="Policy I/O pipeline: policy signature, pipeline hash and tensor pre-processing nodes.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
)
