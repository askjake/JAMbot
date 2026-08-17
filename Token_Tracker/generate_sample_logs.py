#!/usr/bin/env python3
'''
Sample Log Generator for AI Agent Token Monitor
Generates realistic log entries with token usage statistics
'''

import random
import time
from datetime import datetime
import sys

def generate_log_entry(message_num):
    '''Generate a single log entry with random token statistics'''
    
    # Message types with different token patterns
    message_types = ['agent', 'human', 'tool', 'assistant']
    message_type = random.choice(message_types)
    
    # Realistic token ranges based on message type
    token_ranges = {
        'agent': {'input': (800, 2000), 'output': (200, 800), 'reasoning': (100, 500)},
        'human': {'input': (50, 300), 'output': (0, 0), 'reasoning': (0, 0)},
        'tool': {'input': (100, 500), 'output': (50, 300), 'reasoning': (0, 100)},
        'assistant': {'input': (500, 1500), 'output': (300, 1000), 'reasoning': (50, 300)}
    }
    
    ranges = token_ranges[message_type]
    input_tokens = random.randint(*ranges['input'])
    output_tokens = random.randint(*ranges['output'])
    reasoning_tokens = random.randint(*ranges['reasoning'])
    total_tokens = input_tokens + output_tokens + reasoning_tokens
    cache_tokens = random.randint(0, input_tokens // 2) if random.random() > 0.7 else 0
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Various log formats
    formats = [
        f"{timestamp} INFO Message {message_num} - message_type: {message_type}, input_tokens: {input_tokens}, output_tokens: {output_tokens}, reasoning_tokens: {reasoning_tokens}, total_tokens: {total_tokens}",
        f"{timestamp} DEBUG [{message_type.upper()}] input_tokens={input_tokens}, output_tokens={output_tokens}, reasoning_tokens={reasoning_tokens}",
        f"{timestamp} INFO Agent statistics - type={message_type} | input={input_tokens} | output={output_tokens} | reasoning={reasoning_tokens} | cache={cache_tokens}",
    ]
    
    return random.choice(formats)

def main():
    '''Main function to generate sample logs'''
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sample_agent_logs.log'
    continuous = '--continuous' in sys.argv
    delay = 2  # seconds between entries in continuous mode
    
    print(f"📝 Generating sample logs to: {output_file}")
    print(f"🔄 Mode: {'Continuous' if continuous else 'Single batch (20 entries)'}")
    
    try:
        with open(output_file, 'a') as f:
            if continuous:
                print("\n⏸️  Press Ctrl+C to stop\n")
                message_num = 1
                while True:
                    entry = generate_log_entry(message_num)
                    f.write(entry + '\n')
                    f.flush()  # Ensure immediate write
                    print(f"✅ {message_num}: {entry}")
                    message_num += 1
                    time.sleep(delay)
            else:
                print("\nGenerating 20 sample entries...\n")
                for i in range(1, 21):
                    entry = generate_log_entry(i)
                    f.write(entry + '\n')
                    print(f"✅ {i}: {entry}")
                    time.sleep(0.1)  # Small delay to vary timestamps
                
                print(f"\n✨ Done! Generated 20 log entries in {output_file}")
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main()
