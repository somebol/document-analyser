"""${message}"""
from alembic import op
import sqlalchemy as sa
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

# Apply the schema changes for this revision.
def upgrade() -> None:
    ${upgrades if upgrades else "pass"}

# Revert the schema changes for this revision.
def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
