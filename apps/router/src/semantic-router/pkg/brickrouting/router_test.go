package brickrouting

import (
	"testing"

	"github.com/regolo-ai/brick-SR1/apps/router/src/semantic-router/pkg/config"
)

func TestScoreModelsPrefersSufficientModelWithoutOverkill(t *testing.T) {
	r := &Router{
		skillCfg: config.SkillRouterConfig{
			Models: []config.SkillRouterModelConfig{
				{Model: "small", SkillVector: []float64{0.60, 0.60}},
				{Model: "fit", SkillVector: []float64{0.80, 0.80}},
				{Model: "large", SkillVector: []float64{0.95, 0.95}, CostWeight: 1.0},
			},
		},
		mathCfg: newMathConfig(config.SkillRouterMathConfig{}),
	}

	scores := r.scoreModels([]float64{0.5, 0.5}, 0.72)
	if len(scores) != 3 {
		t.Fatalf("expected 3 scores, got %d", len(scores))
	}
	if scores[0].Model != "fit" {
		t.Fatalf("expected fit model, got %s (scores=%+v)", scores[0].Model, scores)
	}
}

func TestKeywordOverrideBeatsBias(t *testing.T) {
	r := &Router{
		skillCfg: config.SkillRouterConfig{
			KeywordRules: []config.SkillRouterKeywordRule{
				{Name: "bias", Mode: "bias", Importance: 8, Capability: "coding", Keywords: []string{"python"}},
				{Name: "override", Mode: "override", Importance: 9, Model: "kimi2.6", Keywords: []string{"debug"}},
			},
		},
	}

	matches := r.matchKeywords("debug this python runtime")
	override := bestOverride(matches)
	if override == nil {
		t.Fatal("expected override match")
	}
	if override.rule.Model != "kimi2.6" {
		t.Fatalf("expected kimi2.6 override, got %s", override.rule.Model)
	}
}
