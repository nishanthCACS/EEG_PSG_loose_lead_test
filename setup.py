from setuptools import setup, find_packages

setup(name='sleep_EEG_loose_lead_detect',
      version='0.0',
      description='Sleep- EEG potential loose-lead detection',
      author = "Nishanth Anandanadarajah",
      author_email='nishyniehs@gmail.com',
      zip_safe=False,
      package_dir = {"": "src"},
      packages =  find_packages(where="src"),
      )
