import type { ChatMessage } from '../client/openai.js';

export interface BuildAgentPromptArgs {
  query: string;
  history: ChatMessage[];
  turn: number;
  selfModel: string;
  previousResponses?: Map<string, string>;
}

export function buildAgentPrompt(args: BuildAgentPromptArgs): ChatMessage[] {
  const { query, history, turn, selfModel, previousResponses } = args;

  if (turn === 1 || !previousResponses || previousResponses.size === 0) {
    return [...history, { role: 'user', content: query }];
  }

  const others = Array.from(previousResponses.entries()).filter(([m]) => m !== selfModel);

  if (others.length === 0) {
    return [...history, { role: 'user', content: query }];
  }

  const othersBlock = others
    .map(([m, r]) => `--- Modello ${m} ---\n${r.trim()}`)
    .join('\n\n');

  const myPrev = previousResponses.get(selfModel);

  const systemAddendum: ChatMessage = {
    role: 'system',
    content:
      `Sei il modello "${selfModel}" in una discussione multi-modello (modalità BABL/Babele).\n` +
      `Stai partecipando al turno ${turn}. Al turno precedente, gli altri modelli hanno risposto alla query dell'utente.\n\n` +
      `RISPOSTE DEGLI ALTRI MODELLI AL TURNO ${turn - 1}:\n\n${othersBlock}\n\n` +
      (myPrev ? `LA TUA RISPOSTA PRECEDENTE:\n${myPrev.trim()}\n\n` : '') +
      `ORA produci una risposta migliorata che:\n` +
      `1. Integri i punti validi delle altre risposte\n` +
      `2. Critichi/corregga eventuali errori che noti\n` +
      `3. Aggiunga prospettive mancanti\n` +
      `Non ripetere semplicemente la tua risposta precedente. Sii conciso e diretto.`,
  };

  return [...history, systemAddendum, { role: 'user', content: query }];
}

export function buildModeratorPrompt(query: string, finalResponses: Map<string, string>): ChatMessage[] {
  const responsesBlock = Array.from(finalResponses.entries())
    .map(([m, r]) => `--- Modello ${m} ---\n${r.trim()}`)
    .join('\n\n');

  return [
    {
      role: 'system',
      content:
        `Sei un moderatore esperto. Hai ricevuto ${finalResponses.size} risposte da modelli LLM diversi alla stessa query dell'utente.\n` +
        `Il tuo compito è sintetizzare in una risposta unica, coerente, accurata e di alta qualità.\n` +
        `Risolvi contraddizioni, combina i punti di forza, ignora errori evidenti.\n` +
        `NON menzionare i singoli modelli o il fatto che ci siano più risposte: produci una risposta finita come se fosse tua.`,
    },
    {
      role: 'user',
      content:
        `Query originale dell'utente:\n"${query}"\n\nRisposte dei modelli:\n\n${responsesBlock}\n\nProduci ora la sintesi finale.`,
    },
  ];
}
