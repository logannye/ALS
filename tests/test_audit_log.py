"""Tests for Task 10: Append-only audit event logger."""

from datetime import datetime, timezone
import pytest


class TestAuditEventDataclass:
    def test_creation_with_required_field(self):
        from audit.event_log import AuditEvent
        evt = AuditEvent(event_type="entity.created")
        assert evt.event_type == "entity.created"

    def test_defaults(self):
        from audit.event_log import AuditEvent
        evt = AuditEvent(event_type="entity.created")
        assert evt.actor == "erik_system"
        assert evt.object_id is None
        assert evt.object_type is None
        assert evt.details == {}
        assert evt.trace_id is None

    def test_created_at_auto_populated(self):
        from audit.event_log import AuditEvent
        evt = AuditEvent(event_type="entity.created")
        assert evt.created_at is not None
        assert isinstance(evt.created_at, datetime)

    def test_all_fields_settable(self):
        from audit.event_log import AuditEvent
        now = datetime.now(timezone.utc)
        evt = AuditEvent(
            event_type="entity.updated",
            object_id="entity:sod1",
            object_type="Entity",
            actor="dr_miller",
            details={"field": "confidence", "old": 0.5, "new": 0.8},
            trace_id="trace-abc-123",
            created_at=now,
        )
        assert evt.event_type == "entity.updated"
        assert evt.object_id == "entity:sod1"
        assert evt.object_type == "Entity"
        assert evt.actor == "dr_miller"
        assert evt.details["field"] == "confidence"
        assert evt.trace_id == "trace-abc-123"
        assert evt.created_at == now

    def test_details_defaults_to_empty_dict(self):
        from audit.event_log import AuditEvent
        evt1 = AuditEvent(event_type="x")
        evt2 = AuditEvent(event_type="y")
        evt1.details["a"] = 1
        # Mutable default must not be shared between instances
        assert "a" not in evt2.details


class TestAuditLoggerModuleInterface:
    def test_audit_logger_importable(self):
        from audit.event_log import AuditLogger  # noqa: F401

    def test_log_method_exists(self):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        assert callable(logger.log)

    def test_query_method_exists(self):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        assert callable(logger.query)

    def test_delete_test_events_method_exists(self):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        assert callable(logger.delete_test_events)


class TestAuditLoggerLive:
    """Live DB tests — require db_available fixture."""

    _TEST_OBJECT_ID = "test:audit_logger_task10"

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_available):
        """Ensure test rows are cleaned up before and after each test."""
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        logger.delete_test_events(self._TEST_OBJECT_ID)
        yield
        logger.delete_test_events(self._TEST_OBJECT_ID)

    def test_log_inserts_event(self, db_available):
        from audit.event_log import AuditLogger, AuditEvent
        logger = AuditLogger()
        evt = logger.log(
            event_type="entity.created",
            object_id=self._TEST_OBJECT_ID,
            object_type="Entity",
            actor="test_suite",
            details={"name": "SOD1", "confidence": 0.9},
            trace_id="trace-test-001",
        )
        assert isinstance(evt, AuditEvent)
        assert evt.event_type == "entity.created"
        assert evt.object_id == self._TEST_OBJECT_ID
        assert evt.created_at is not None

    def test_query_returns_logged_event(self, db_available):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        logger.log(
            event_type="entity.created",
            object_id=self._TEST_OBJECT_ID,
            object_type="Entity",
            actor="test_suite",
            details={"name": "SOD1"},
        )
        events = logger.query(object_id=self._TEST_OBJECT_ID)
        assert len(events) == 1
        assert events[0].event_type == "entity.created"

    def test_query_verifies_all_fields(self, db_available):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        logger.log(
            event_type="entity.updated",
            object_id=self._TEST_OBJECT_ID,
            object_type="Entity",
            actor="dr_test",
            details={"field": "confidence", "value": 0.75},
            trace_id="trace-verify-001",
        )
        events = logger.query(object_id=self._TEST_OBJECT_ID)
        assert len(events) == 1
        evt = events[0]
        assert evt.event_type == "entity.updated"
        assert evt.object_id == self._TEST_OBJECT_ID
        assert evt.object_type == "Entity"
        assert evt.actor == "dr_test"
        assert evt.details["field"] == "confidence"
        assert evt.details["value"] == 0.75
        assert evt.trace_id == "trace-verify-001"
        assert isinstance(evt.created_at, datetime)

    def test_query_filter_by_event_type(self, db_available):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        logger.log(event_type="entity.created", object_id=self._TEST_OBJECT_ID)
        logger.log(event_type="entity.deleted", object_id=self._TEST_OBJECT_ID)
        created_events = logger.query(
            object_id=self._TEST_OBJECT_ID, event_type="entity.created"
        )
        assert len(created_events) == 1
        assert created_events[0].event_type == "entity.created"

    def test_query_respects_limit(self, db_available):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        for i in range(5):
            logger.log(event_type="entity.created", object_id=self._TEST_OBJECT_ID)
        events = logger.query(object_id=self._TEST_OBJECT_ID, limit=3)
        assert len(events) == 3

    def test_delete_test_events_cleans_up(self, db_available):
        from audit.event_log import AuditLogger
        logger = AuditLogger()
        logger.log(event_type="entity.created", object_id=self._TEST_OBJECT_ID)
        deleted = logger.delete_test_events(self._TEST_OBJECT_ID)
        assert deleted >= 1
        remaining = logger.query(object_id=self._TEST_OBJECT_ID)
        assert remaining == []
