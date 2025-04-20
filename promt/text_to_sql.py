import ollama
import sqlite3
from sql_metadata import Parser
import re
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from typing import List, Dict, Tuple


class TextToSQLEvaluator:
    def __init__(self, model1_name: str, model2_name: str, database_path: str = None):
        """
        Initialize the evaluator with two Ollama models and an optional database for validation.

        Args:
            model1_name: Name of the first Ollama model
            model2_name: Name of the second Ollama model
            database_path: Path to SQLite database for validation (optional)
        """
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.database_path = database_path
        self.results = []

        # Define the schema and prompt template
        self.schema = """
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id
"""

        # Test connection to Ollama
        try:
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

        if database_path:
            try:
                self.conn = sqlite3.connect(database_path)
                self.cursor = self.conn.cursor()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to database: {e}")

    def generate_sql(self, question: str, model_name: str) -> str:
        """
        Generate SQL from natural language question using specified model with your custom prompt format.

        Args:
            question: Natural language question
            model_name: Which model to use

        Returns:
            Generated SQL query
        """
        prompt_template = f"""
### Instructions:
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{self.schema}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
"""
        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt_template,
                stream=False
            )
            # Extract SQL from between ```sql ``` markers
            sql = response['response'].strip()
            match = re.search(r'```sql\n(.*?)\n```', sql, re.DOTALL)
            if match:
                return match.group(1).strip()
            return sql
        except Exception as e:
            print(f"Error generating SQL with {model_name}: {e}")
            return ""

    def validate_sql_syntax(self, query: str) -> bool:
        """Check if SQL query has valid syntax."""
        if not query:
            return False

        try:
            Parser(query)
            return True
        except:
            return False

    def validate_sql_execution(self, query: str) -> bool:
        """Execute SQL query against the database to validate it works."""
        if not self.database_path or not query:
            return False

        try:
            if any(keyword in query.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE']):
                return False

            self.cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            return True
        except:
            return False

    def check_aliases_usage(self, query: str) -> bool:
        """Check if query uses table aliases as specified in requirements."""
        if not query:
            return False

        try:
            parser = Parser(query)
            tables = parser.tables

            # Check if all tables in FROM/JOIN clauses have aliases
            from_match = re.search(r'FROM\s+([^\s,(]+)(?:\s+AS\s+|\s+)([^\s,)]+)', query, re.IGNORECASE)
            join_matches = re.finditer(r'JOIN\s+([^\s,(]+)(?:\s+AS\s+|\s+)([^\s,)]+)', query, re.IGNORECASE)

            # Check if columns are referenced with table aliases
            column_refs = re.finditer(
                r'(\b(?:SELECT|WHERE|GROUP BY|HAVING|ORDER BY)\b.*?)(\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)',
                query, re.IGNORECASE)

            return bool(from_match) and any(join_matches) and any(column_refs)
        except:
            return False

    def check_float_casting(self, query: str) -> bool:
        """Check if ratios are properly cast as float."""
        if not query:
            return False

        # Look for division operations with float casting
        return bool(re.search(r'CAST\(.*?AS\s+FLOAT\)\s*\/', query, re.IGNORECASE)) or \
            bool(re.search(r'\/.*?CAST\(.*?AS\s+FLOAT\)', query, re.IGNORECASE))

    def evaluate_prompt(self, question: str, prompt_type: str = 'general') -> dict:
        """
        Evaluate both models on a single question with your custom requirements.

        Args:
            question: Natural language question
            prompt_type: Category of query (e.g., 'simple', 'join', 'aggregate')

        Returns:
            Dictionary with evaluation results
        """
        # Generate SQL from both models
        start_time1 = time.time()
        sql1 = self.generate_sql(question, self.model1_name)
        time1 = time.time() - start_time1

        start_time2 = time.time()
        sql2 = self.generate_sql(question, self.model2_name)
        time2 = time.time() - start_time2

        # Validate syntax
        valid_syntax1 = self.validate_sql_syntax(sql1)
        valid_syntax2 = self.validate_sql_syntax(sql2)

        # Validate execution if database is available
        valid_execution1 = self.validate_sql_execution(sql1) if valid_syntax1 else False
        valid_execution2 = self.validate_sql_execution(sql2) if valid_syntax2 else False

        # Check custom requirements
        aliases_used1 = self.check_aliases_usage(sql1) if valid_syntax1 else False
        aliases_used2 = self.check_aliases_usage(sql2) if valid_syntax2 else False

        float_casting1 = self.check_float_casting(sql1) if valid_syntax1 else False
        float_casting2 = self.check_float_casting(sql2) if valid_syntax2 else False

        result = {
            'question': question,
            'prompt_type': prompt_type,
            'model1': self.model1_name,
            'model2': self.model2_name,
            'sql1': sql1,
            'sql2': sql2,
            'time1': time1,
            'time2': time2,
            'valid_syntax1': valid_syntax1,
            'valid_syntax2': valid_syntax2,
            'valid_execution1': valid_execution1,
            'valid_execution2': valid_execution2,
            'aliases_used1': aliases_used1,
            'aliases_used2': aliases_used2,
            'float_casting1': float_casting1,
            'float_casting2': float_casting2
        }

        self.results.append(result)
        return result

    def evaluate_questions(self, questions: List[Tuple[str, str]]) -> List[dict]:
        """Evaluate multiple questions."""
        return [self.evaluate_prompt(question, prompt_type) for question, prompt_type in questions]

    def calculate_metrics(self) -> dict:
        """Calculate aggregate metrics across all evaluations."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        metrics = {
            'total_questions': len(df),
            'model1_syntax_accuracy': df['valid_syntax1'].mean(),
            'model2_syntax_accuracy': df['valid_syntax2'].mean(),
            'model1_execution_accuracy': df['valid_execution1'].mean(),
            'model2_execution_accuracy': df['valid_execution2'].mean(),
            'model1_aliases_usage': df['aliases_used1'].mean(),
            'model2_aliases_usage': df['aliases_used2'].mean(),
            'model1_float_casting': df['float_casting1'].mean(),
            'model2_float_casting': df['float_casting2'].mean(),
            'model1_avg_time': df['time1'].mean(),
            'model2_avg_time': df['time2'].mean()
        }

        # Calculate accuracy by question type
        for prompt_type in df['prompt_type'].unique():
            type_df = df[df['prompt_type'] == prompt_type]
            metrics[f'model1_syntax_accuracy_{prompt_type}'] = type_df['valid_syntax1'].mean()
            metrics[f'model2_syntax_accuracy_{prompt_type}'] = type_df['valid_syntax2'].mean()
            metrics[f'model1_execution_accuracy_{prompt_type}'] = type_df['valid_execution1'].mean()
            metrics[f'model2_execution_accuracy_{prompt_type}'] = type_df['valid_execution2'].mean()

        return metrics

    def get_recommendation(self) -> dict:
        """Get recommendation on which model to use based on evaluation results."""
        metrics = self.calculate_metrics()

        if not metrics:
            return {'recommendation': 'Insufficient data', 'reason': 'No evaluations performed'}

        # Calculate scores with weights for custom requirements
        model1_score = (
                0.3 * metrics['model1_syntax_accuracy'] +
                0.3 * metrics['model1_execution_accuracy'] +
                0.2 * metrics['model1_aliases_usage'] +
                0.1 * metrics['model1_float_casting'] +
                0.1 * (1 - metrics['model1_avg_time'] / max(metrics['model1_avg_time'], metrics['model2_avg_time']))
        )

        model2_score = (
                0.3 * metrics['model2_syntax_accuracy'] +
                0.3 * metrics['model2_execution_accuracy'] +
                0.2 * metrics['model2_aliases_usage'] +
                0.1 * metrics['model2_float_casting'] +
                0.1 * (1 - metrics['model2_avg_time'] / max(metrics['model1_avg_time'], metrics['model2_avg_time']))
        )

        recommendation = {
            'model1_score': model1_score,
            'model2_score': model2_score,
            'recommendation': self.model1_name if model1_score > model2_score else self.model2_name,
            'reason': (
                f"Recommended {self.model1_name if model1_score > model2_score else self.model2_name} "
                f"based on higher overall score ({max(model1_score, model2_score):.2f} vs {min(model1_score, model2_score):.2f}).\n"
                f"Model 1: Syntax={metrics['model1_syntax_accuracy']:.2f}, Execution={metrics['model1_execution_accuracy']:.2f}, "
                f"Aliases={metrics['model1_aliases_usage']:.2f}, FloatCast={metrics['model1_float_casting']:.2f}\n"
                f"Model 2: Syntax={metrics['model2_syntax_accuracy']:.2f}, Execution={metrics['model2_execution_accuracy']:.2f}, "
                f"Aliases={metrics['model2_aliases_usage']:.2f}, FloatCast={metrics['model2_float_casting']:.2f}"
            )
        }

        # Add strengths for each model by question type
        strengths = {}
        for prompt_type in set([r['prompt_type'] for r in self.results]):
            model1_acc = metrics.get(f'model1_syntax_accuracy_{prompt_type}', 0)
            model2_acc = metrics.get(f'model2_syntax_accuracy_{prompt_type}', 0)

            if model1_acc > model2_acc:
                strengths[f'model1_better_at_{prompt_type}'] = model1_acc - model2_acc
            else:
                strengths[f'model2_better_at_{prompt_type}'] = model2_acc - model1_acc

        recommendation['strengths'] = strengths
        return recommendation

    def save_results(self, file_path: str = 'text_to_sql_evaluation_results.csv'):
        """Save evaluation results to CSV file."""
        pd.DataFrame(self.results).to_csv(file_path, index=False)


# Example Usage
if __name__ == "__main__":
    # Initialize evaluator with your models of choice
    evaluator = TextToSQLEvaluator(
        model1_name="sqlcoder",  # First model to compare
        model2_name="llama3.2",  # Second model to compare
        database_path="sales_database.db"  # Optional, omit if not available
    )

    # Define test questions with their types
    test_questions = [
        ("List all products with quantity less than 50", "simple"),
        ("Show total sales by product category", "aggregate"),
        ("Find customers who purchased more than 5 items in a single order", "complex"),
        ("Which salesperson has the highest total sales amount?", "aggregate"),
        ("What is the ratio of products with low stock (quantity < 20) to total products?", "ratio"),
        ("List all sales with product details and customer information", "join"),
        ("Show the average sale price by region", "aggregate_join"),
        ("Which products have never been sold?", "complex_join"),
    ]

    # Run evaluations
    evaluator.evaluate_questions(test_questions)

    # Get metrics and recommendation
    metrics = evaluator.calculate_metrics()
    recommendation = evaluator.get_recommendation()

    # Print results
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print("\nRecommendation:")
    print(recommendation['reason'])
    print("\nModel Strengths by Question Type:")
    for k, v in recommendation['strengths'].items():
        print(f"{k}: {v:.2f}")

    # Save results
    evaluator.save_results()
