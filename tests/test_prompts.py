import pytest
from prompts.genome import PromptGenotype, PromptExample
from prompts.store import PromptGenomeStore
from prompts.mutations import MutationEngine, MutationOperator
from prompts.crossover import CrossoverEngine
from prompts.fitness import PromptFitnessTracker, PopulationManager

class TestPromptsModule:
    """Tests for the Prompt Genome System."""
    
    @pytest.fixture
    def sample_genome(self):
        return PromptGenotype(
            id="test_genome_001",
            instruction_block="Extract PII",
            entity_definitions={"PERSON": "Names of people"},
            examples=[PromptExample(input_text="My name is John", expected_output='[{"text": "John", "label": "PERSON"}]', entity_type="PERSON")],
            constraints=["Be accurate"]
        )

    def test_genome_creation(self, sample_genome):
        assert sample_genome.instruction_block == "Extract PII"
        assert len(sample_genome.examples) == 1

    def test_mutation_engine(self, sample_genome):
        engine = MutationEngine()
        mutated = engine.apply_mutation(sample_genome, MutationOperator.ADD_CONSTRAINT, {"constraint": "New constraint"})
        assert len(mutated.constraints) == len(sample_genome.constraints) + 1
        assert "New constraint" in mutated.constraints

    def test_genome_store(self, tmp_path):
        db_path = tmp_path / "prompts.db"
        store = PromptGenomeStore(str(db_path))
        genome = PromptGenotype.create_default()
        store.save(genome)
        loaded = store.get(genome.id)
        assert loaded.id == genome.id
