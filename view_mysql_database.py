import mysql.connector
import pandas as pd
from tabulate import tabulate

# Database configuration
config = {
    'user': 'root',
    'password': 'password',
    'host': 'localhost',
    'database': 'student_performance'
}

def execute_query(query):
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(results, columns=columns)
        
        # Close connection
        cursor.close()
        conn.close()
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # List all tables
    print("\nDatabase Tables:")
    tables_query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'student_performance'
    """
    tables_df = execute_query(tables_query)
    print(tables_df)
    
    # For each table, show its structure and data
    for table in tables_df['table_name']:
        print(f"\nTable: {table}")
        
        # Show table structure
        print("\nTable Structure:")
        structure_query = f"""
        DESCRIBE {table}
        """
        structure_df = execute_query(structure_query)
        print(tabulate(structure_df, headers='keys', tablefmt='psql'))
        
        # Show table data
        print(f"\nData in {table}:")
        data_query = f"""
        SELECT * FROM {table}
        """
        data_df = execute_query(data_query)
        if not data_df.empty:
            print(tabulate(data_df, headers='keys', tablefmt='psql'))
        else:
            print("No data found")

if __name__ == "__main__":
    main() 