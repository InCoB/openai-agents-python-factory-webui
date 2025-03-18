"""Tests for template loading and rendering."""

import os
import pytest
from jinja2 import exceptions

from aigen.services.generator import AgentGeneratorService
from aigen.services.models import AgentConfiguration, AgentRole


def test_template_loading():
    """Test that templates can be loaded correctly."""
    generator = AgentGeneratorService()

    # Check template files exist
    template_files = ["agent_class.py.jinja", "agent_config.yaml.jinja"]
    for filename in template_files:
        template_path = os.path.join(generator.template_dir, filename)
        assert os.path.exists(template_path), f"Template file {filename} does not exist"

    # Try loading templates through Jinja
    for filename in template_files:
        try:
            template = generator.env.get_template(filename)
            assert template is not None, f"Failed to load template: {filename}"
        except exceptions.TemplateNotFound:
            pytest.fail(f"Template {filename} not found by Jinja environment")


def test_template_rendering():
    """Test that templates can be rendered with sample data."""
    generator = AgentGeneratorService()

    # Create sample configuration
    config = AgentConfiguration(
        agent_type="test_agent",
        name="Test Agent",
        role=AgentRole.CUSTOM,
        instructions="You are a test agent for verifying template rendering.",
        model="gpt-4o",
        parameters={"temperature": 0.5, "max_tokens": 2000},
        tools=["calculator", "web_search"],
        handoffs=["another_agent"],
    )

    # Test Python code generation
    success, code = generator.generate_agent_code(config)
    assert success, "Failed to generate agent code"
    assert (
        "class TestAgentAgent:" in code
    ), "Generated code doesn't contain expected class definition"

    # Test YAML generation
    success, yaml_config = generator.generate_yaml_config(config)
    assert success, "Failed to generate YAML config"
    assert (
        "agent_type: test_agent" in yaml_config
    ), "Generated YAML doesn't contain expected content"


if __name__ == "__main__":
    test_template_loading()
    test_template_rendering()
    print("✅ All template tests passed!")
