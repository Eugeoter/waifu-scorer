from setuptools import setup, find_packages
with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

for i, req in enumerate(requirements):
    if req.startswith('git+'):
        package_name = req.split('/')[-1].split('.')[0]  # Extract package name from URL
        requirements[i] = f"{package_name} @ {req}"

setup(
    name='waifu-scorer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Image caption tools',
    long_description='',
    author='euge',
    author_email='1507064225@qq.com',
    url='https://github.com/Eugeoter/waifu-scorer',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
