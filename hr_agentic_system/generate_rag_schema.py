from sqlalchemy import create_engine, inspect
import json

def sqlite_to_rag_json_schema(db_path: str, output_path: str = None) -> dict:
    """
    Converts a SQLite database into a RAG-friendly JSON schema format.

    Args:
        db_path (str): Path to the SQLite database file.
        output_path (str): Optional path to save the output JSON file.

    Returns:
        dict: Dictionary representing the schema of the database.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    schema = {}

    for table_name in inspector.get_table_names():
        table_info = {
            "columns": [],
            "primary_keys": inspector.get_pk_constraint(table_name).get("constrained_columns", []),
            "foreign_keys": [],
        }

        for column in inspector.get_columns(table_name):
            column_info = {
                "name": column["name"],
                "type": str(column["type"]),
            }
            table_info["columns"].append(column_info)

        for fk in inspector.get_foreign_keys(table_name):
            fk_info = {
                "column": fk["constrained_columns"][0],
                "ref_table": fk["referred_table"],
                "ref_column": fk["referred_columns"][0],
            }
            table_info["foreign_keys"].append(fk_info)

        schema[table_name] = table_info

    if output_path:
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)

    return schema

# Example usage
if __name__ == "__main__":
    db_path = "hr_system.db"  # path to your SQLite file
    output_path = "rag_schema.json"
    schema = sqlite_to_rag_json_schema(db_path, output_path)
    print(json.dumps(schema, indent=2))
