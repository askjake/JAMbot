# Database Tables

## User
### User Global State Table `user_state`
| `Name` | `Type` | `Nullable` | `Default` | `Comment` |
|--------|--------|------------|-----------|-----------|
| user_email | text | False | | PK |
| current_chat_id | uuid |  True | null | FK(chat.chat_id) ON DELETE SET NULL |

## Chat
### User Chat Table `chat`
| `Name` | `Type` | `Nullable` | `Default` | `Comment` |
|--------|--------|------------|-----------|-----------|
| chat_id | uuid | False | | PK |
| title | text | False | | |
| created_at | timestamptz | False | now() | |
| owner_id | text | False | | |
| last_message_at | text | False | now() | Initially same as created_at|
| vault_mode | boolean | False | | |
| favorite | boolean | False | | |



## Message
### Message Table `message`
| `Name` | `Type` | `Nullable` | `Default` | `Comment` |
|--------|--------|------------|-----------|-----------|
| message_id | uuid | False | | PK |
| chat_id | uuid | False | | FK on chat.chat_id |
| role | enum('user', 'ai', 'tool') | False | | |
| checkpoint_id | str | False | | |
| version_id | int | False | | PK |
| parent | uuid | False | | |

## Attachment
### Attachment Table `attachment`
| `Name` | `Type` | `Nullable` | `Default` | `Comment` |
|--------|--------|------------|-----------|-----------|
| id | uuid | False | | PK |
| filename | text | False | | |
| owner_id | text | False | | |
| chat_id | uuid | False | | FK on `chat.chat_id` |
| media_type | text | False | | |
| created_at | timestamptz | False | now() | |
| updated_at | timestamptz | False | now() | |
| status | enum | False | "queued" | |
| vault_mode | bool | False | | |
| s3_location | text | False | | |
| progress | int | False | 0 | |
| Progress_msg | text | False | "" | |

## Vault
### Vault Session Table `vault_session`
| `Name` | `Type` | `Nullable` | `Default` | `Comment` |
|--------|--------|------------|-----------|-----------|
| user | text | False | | PK |
