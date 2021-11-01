# E2EWatch: An End-to-End Anomaly Diagnosis Framework for Production HPC Systems

Deployment codes for Anomaly Diagnosis Framework 

__author__ & __maintainer__ = "Burak Aksar"

__email__ = "baksar@bu.edu"

__version__ = "1.0.0"



## __Install Virtual Environment__:

1-) Create a local virtual environment in the folder

`python3 -m venv ml_venv`

2-) Activate venv

`source ml_venv/bin/activate/`

3-) Install requirements

`pip install -r requirements.txt`

## __Running__:

Run the jupyter notebook inside the venv, not in your local

`./ml_venv/bin/jupyter notebook`

Under the analysis folder you will find necessary scripts to replicate unknown apps, unknown inputs and the defauly anomaly diagnosis experiments. 

The predict.py can be used to train a model and then you can use the RuntimePredictor class under runtime folder. 

At a high level, E2EWatch requires the following components to provide diagnosis results at runtime in another production system: 

* Monitoring framework that can collect numeric telemetry data from compute nodes while applications are running. Even though we only experiment with LDMS, it can be adapted to other popular monitoring frameworks such as Ganglia, Examon by modifying the wrappers in the data collection phase. 

* Labeled data that is composed of anomalous and normal compute node telemetry data. It is possible to create labeled data sets using a suite of applications and synthetic anomalies. Another option is to use telemetry data labeled by users. 

* Backend web service that can provide telemetry data on the fly to the trained model. We use the existing Django web application deployed on the [monitoring server](https://ieeexplore.ieee.org/document/9229587). It is possible to use other backend web services that can handle client requests and query data from the database. If runtime diagnosis is not necessary, it is also possible to run the pickled model after the application run is completed. 


## Authors

[E2EWatch: An End-to-End Anomaly Diagnosis Framework for Production HPC System](https://link.springer.com/chapter/10.1007/978-3-030-85665-6_5)


Authors:
    Burak Aksar (1), Benjamin Schwaller (2), Omar Aaziz (2), Vitus J. Leung (2), Jim Brandt (2), Manuel Egele (1), Ayse K. Coskun (1)

Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details



