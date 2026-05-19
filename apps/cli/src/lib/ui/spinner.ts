import chalk from 'chalk';

const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const FRAME_MS = 80;

export class Spinner {
  private timer: NodeJS.Timeout | null = null;
  private idx = 0;
  private label = '';
  private active = false;
  private isTty = !!process.stdout.isTTY;

  start(label: string): void {
    this.label = label;
    this.active = true;
    this.idx = 0;
    if (!this.isTty) return;
    this.render();
    this.timer = setInterval(() => {
      this.idx = (this.idx + 1) % FRAMES.length;
      this.render();
    }, FRAME_MS);
  }

  update(label: string): void {
    this.label = label;
    if (this.active && this.isTty) this.render();
  }

  succeed(label: string): void {
    this.stop();
    process.stdout.write(chalk.green('✓ ') + chalk.green(label) + '\n');
  }

  warn(label: string): void {
    this.stop();
    process.stdout.write(chalk.yellow('⚠ ') + chalk.yellow(label) + '\n');
  }

  fail(label: string): void {
    this.stop();
    process.stdout.write(chalk.red('✗ ') + chalk.red(label) + '\n');
  }

  /** Clear spinner line (without printing anything else). Use when restoring foreground UI. */
  clear(): void {
    this.stop();
  }

  isActive(): boolean { return this.active; }

  private stop(): void {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
    if (this.active && this.isTty) {
      // clear current line
      process.stdout.write('\r\x1b[2K');
    }
    this.active = false;
  }

  private render(): void {
    process.stdout.write(`\r\x1b[2K${chalk.green(FRAMES[this.idx])} ${chalk.green(this.label)}`);
  }
}
