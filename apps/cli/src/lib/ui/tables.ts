import Table from 'cli-table3';

export const LEFT_PAD = '  ';

export function makeTable(head: string[]): Table.Table {
  return new Table({ head, style: { head: ['cyan'] } });
}

/** Render a cli-table3 with each line left-padded for terminal margin. */
export function renderTable(table: Table.Table): string {
  return table.toString().split('\n').map((l) => LEFT_PAD + l).join('\n');
}

/** Indent every line of `text` with the standard left padding. */
export function pad(text: string): string {
  return text.split('\n').map((l) => LEFT_PAD + l).join('\n');
}
