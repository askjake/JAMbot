"""add user journal table

Revision ID: journal_001
Revises: 
Create Date: 2026-01-24

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = 'journal_001'
down_revision = '20260123_add_agent_mode_run'  # Update this to your latest migration
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'user_journal',
        sa.Column('journal_id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('owner_id', sa.Text(), nullable=False),
        sa.Column('chat_id', UUID(as_uuid=True), nullable=True),
        sa.Column('journal_type', sa.String(50), nullable=False),
        
        # Journal Content
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('psychoanalysis', JSONB, nullable=True),
        sa.Column('interaction_patterns', JSONB, nullable=True),
        sa.Column('user_preferences', JSONB, nullable=True),
        sa.Column('topics', JSONB, nullable=True),
        sa.Column('sentiment_analysis', JSONB, nullable=True),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('conversation_start', sa.DateTime(), nullable=True),
        sa.Column('conversation_end', sa.DateTime(), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=True),
        
        # Status
        sa.Column('status', sa.String(20), server_default='active', nullable=False),
        
        sa.ForeignKeyConstraint(['owner_id'], ['user_state.owner_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['chat_id'], ['chat.chat_id'], ondelete='SET NULL'),
    )
    
    # Create indexes
    op.create_index('idx_journal_owner', 'user_journal', ['owner_id', 'created_at'])
    op.create_index('idx_journal_chat', 'user_journal', ['chat_id'])
    op.create_index('idx_journal_type', 'user_journal', ['journal_type'])


def downgrade():
    op.drop_index('idx_journal_type', table_name='user_journal')
    op.drop_index('idx_journal_chat', table_name='user_journal')
    op.drop_index('idx_journal_owner', table_name='user_journal')
    op.drop_table('user_journal')
