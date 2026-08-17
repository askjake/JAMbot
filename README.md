# Dish-Chat Backend

## Project Setup for Development Environment

Follow these steps to set up the development environment for the `dish-chat` backend project:

### Prerequisites
1. **Install Required Tools**:
   - [Docker](https://www.docker.com/) and Docker Compose.
   - Python 3.12 (as specified in `.python-version`).
   - `pip` and `virtualenv` or equivalent for Python dependency management.
2. **AWS Configuration**:
   - Ensure your local AWS environment is configured with the necessary credentials (SecGateway)


### Steps to Set Up 

1. **Start local PostgreSQL Database**  
   Use Docker Compose to create and start the `dev_postgres` container:
   ```bash
   cd dev_postgres
   docker compose up -d
   ```

2. **Set Up a Virtual Environment**  
   Create and activate a Python virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**  
   Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Database Migrations**  
   Use Alembic to apply database migrations:
   ```bash
   cd app/
   alembic upgrade head
   ```

5. **Start the Application**  
   Run the application using Uvicorn:
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Verify the Setup**  
   - The application will be available at `http://127.0.0.1:8000`.
   - Check the logs of the `dev_postgres` container if needed:
     ```bash
     docker logs postgres-dev-dishchat
     ```

### Additional Notes
- The debug flag is on by default. It can be adjusted in `app/config.py`
- The database connection URL is configured in `app/config.py` and dynamically set in the Alembic environment (`app/alembic/env.py`).
- For more information about Alembic migrations, refer to the [Alembic documentation](https://alembic.sqlalchemy.org/).
- If you encounter issues with AWS-related functionality, ensure your AWS credentials are valid and have the necessary permissions with SecGateway.

