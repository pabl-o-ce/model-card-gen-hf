# Hugging Face: model card generator

This project is a model card generator for Hugging Face, utilizing Jinja2 and Python.

## Description

Model cards are a useful tool for providing clear, concise information about machine learning models. This project aims to simplify the process of creating model cards for Hugging Face models by using Jinja2 templates and Python scripts.

With this project, you can quickly generate a professional-looking model card that includes information such as the model's architecture, performance metrics, and training data.

## Getting Started

### Dependencies

* Python 3.10 or later
* Jinja2 3.1.x or later

### Installing

* Clone the repository
* Install the required packages using pip: `pip install .`

### Executing program

* To generate a model card, you'll need to provide information about your Hugging Face model. This can be done by creating a YAML file that contains the necessary information.
* An example YAML file is provided in the repository, which you can use as a template.
* Once you've created your YAML file, you can generate your model card by running the following command: `python generate_card.py`
* The output file will be a HTML file, you can open it in your browser to view your model card.

## Help

If you encounter any problems or have any questions, please open an issue in the repository.

## Authors

Pablo Carrera

## License

This project is licensed under the [MIT License](LICENSE.md)

## Acknowledgments

* Inspiration from Hugging Face and their model cards
* Jinja2 for providing the templating engine
* The Python community for their support and contributions

## Project status
The project is currently in active development. We are working on adding more features and improving the overall user experience. If you have any suggestions or feedback, please don't hesitate to let us know.

## Roadmap
Here is a rough outline of the features that we plan to add in the near future:

* Support for more types of models and data
* More customization options for the model cards
* A web-based interface for the model card generator
* Integration with other machine learning platforms and tools.