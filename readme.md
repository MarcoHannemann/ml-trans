#ml-trans

---

This repository contains the source code for the hybrid Priestley-Taylor to predict transpiration, trained with sap flow
observations from the global SAPFLUXNET database.

##How to Install the Project

---

Clone the repository to your local drive with

`git clone https://gitlab.com/mhannemann1/ml-trans.git`

Make sure you have Python 3 installed. If not, you can download it at [https://www.python.org/downloads/](https://www.python.org/downloads/).
This version is tested for Python 3.9. If you want to use a newer release of Python, some functionality could break.

Create a new virtual environment from your terminal e.g. with

```$ python -m venv venv/```

Install the dependencies from the requirements.txt using pip

```$ pip install -r requirements.txt```


##How to use the Project

---

You can use this model to make predictions by running the [tutorial](tutorial.ipynb) Jupyter Notebook-
The Notebook also contains information on how to prepare your own input data to estimate transpiration.



If you want to experiment by training a new model, edit the [configuration](config/config.ini) file with your favourite
text editor. Make sure you have the input data referred to in the `[PATHS]` section. If you have trouble configurating the
model contact the author of this project [marco.hannemann@ufz.de](mailto:marco.hannemann@ufz.de?subject=[GitLab]Support%20with%20ml-trans)

##How to contribute

---

Contributions to the project are welcome. If you find an error, please [raise an issue](https://gitlab.com/mhannemann1/ml-trans/-/issues/new).
If you feel you can solve the error on your own, you are invited to create a merge request.


##References

---

A detailed description of the method can be obtained from Hannemann et al. 2023
