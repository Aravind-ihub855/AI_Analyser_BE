import os
import logging
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceAuthenticationFailed

logger = logging.getLogger(__name__)

def get_salesforce_client():
    """
    Initializes and returns a Salesforce client.
    Supports environment variables SALESFORCE_* or SF_*.
    """
    username = os.getenv("SALESFORCE_USERNAME") or os.getenv("SF_USERNAME")
    password = os.getenv("SALESFORCE_PASSWORD") or os.getenv("SF_PASSWORD")
    security_token = os.getenv("SALESFORCE_SECURITY_TOKEN") or os.getenv("SF_SECURITY_TOKEN")
    domain = os.getenv("SALESFORCE_DOMAIN") or os.getenv("SF_DOMAIN") or "login"
    
    if not all([username, password, security_token]):
        logger.error("[SALESFORCE] Connection failed: Credentials are not fully set in .env")
        return None
        
    try:
        sf = Salesforce(
            username=username,
            password=password,
            security_token=security_token,
            domain=domain
        )
        logger.info(f"[SALESFORCE] Connection successful. Logged in as: {username}")
        return sf
    except SalesforceAuthenticationFailed as e:
        logger.error(f"[SALESFORCE] Authentication Failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"[SALESFORCE] Unexpected connection error: {str(e)}")
        return None

def get_object_fields(object_name: str) -> list:
    """
    Fetches all fields, their data types, and labels for a given Salesforce object.
    Used by the AI to learn object schema dynamically.
    """
    sf = get_salesforce_client()
    if not sf:
        logger.error(f"[SALESFORCE] Cannot retrieve fields: Salesforce client not initialized.")
        return []
    try:
        logger.info(f"[SALESFORCE] Retrieving metadata description for object: {object_name}")
        desc = getattr(sf, object_name).describe()
        fields_info = []
        for f in desc.get("fields", []):
            fields_info.append({
                "name": f.get("name"),
                "type": f.get("type"),
                "label": f.get("label")
            })
        return fields_info
    except Exception as e:
        logger.error(f"[SALESFORCE] Describe object '{object_name}' failed: {str(e)}")
        return []

def query_salesforce(soql_query: str) -> list:
    """
    Executes a SOQL query against Salesforce and returns raw results.
    """
    sf = get_salesforce_client()
    if not sf:
        raise ValueError("Salesforce connection could not be established. Check .env credentials.")
        
    try:
        logger.info(f"[SALESFORCE] Executing SOQL query: {soql_query}")
        result = sf.query_all(soql_query)
        records = result.get("records", [])
        logger.info(f"[SALESFORCE] Query successfully returned {len(records)} records.")
        return records
    except Exception as e:
        logger.error(f"[SALESFORCE] SOQL query execution failed: {str(e)}")
        raise e
