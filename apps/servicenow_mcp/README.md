# ServiceNow ITSM MCP Server

A FastMCP server exposing ServiceNow incident, change, and user search as MCP tools.
Integrates with Jakes-agent via streamable_http transport on port 8095.

## Tools

| Tool | Description |
|------|-------------|
| `snow_search_incidents` | Search incidents by keyword |
| `snow_get_incident` | Fetch a single incident by INC number |
| `snow_search_changes` | Search change requests by keyword |
| `snow_get_change` | Fetch a single change request by CHG number |
| `snow_search_users` | Search users by name or email |
| `snow_query_table` | Generic ServiceNow table query |

## Auth Modes

- **oauth**: OAuth 2.0 Client Credentials (`SNOW_CLIENT_ID`, `SNOW_CLIENT_SECRET`)
- **basic**: HTTP Basic Auth (`SNOW_USERNAME`, `SNOW_PASSWORD`)

## Quick Start

```bash
# Set credentials
export SNOW_INSTANCE_URL=https://dish.service-now.com
export SNOW_AUTH_MODE=oauth
export SNOW_CLIENT_ID=...
export SNOW_CLIENT_SECRET=...

# Start server
bash apps/servicenow_mcp/start-mcp-server.sh
```

## Config

See `config/.env.example` for all environment variables.
