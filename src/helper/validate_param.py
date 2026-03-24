"""Validating a JSON file against a given schema."""
import json
import pathlib
from jsonschema import validate
from jsonschema.exceptions import ValidationError

def validate_json_schema(json_path: pathlib.Path, schema_path : pathlib.Path):
    """Validating a JSON file with the given schema.

    Args:
        json_path (pathlib.Path): path to the JSON file
        schema_path (pathlib.Path): path to the schema file
    """
    with open(json_path) as json_file:
        with open(schema_path) as schema_file:
            json_contents = json.load(json_file)
            schema_contents = json.load(schema_file)

            try:
                validate(instance=json_contents, schema=schema_contents)
            except ValidationError as e:
                print("JSON validation failed!")
