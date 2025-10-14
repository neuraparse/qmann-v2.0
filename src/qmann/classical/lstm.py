"""
Classical LSTM Implementation

Enhanced LSTM with memory-augmented capabilities for integration
with quantum memory systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..core.base import ClassicalComponent
from ..core.memory import ClassicalMemoryInterface


class ClassicalLSTM(ClassicalComponent):
    """
    Enhanced LSTM with memory augmentation capabilities.
    
    Provides classical memory operations that can be seamlessly
    integrated with quantum memory systems.
    """
    
    def __init__(
        self,
        config,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        memory_size: int = 1024,
        name: str = "ClassicalLSTM"
    ):
        super().__init__(config, name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_size = memory_size
        
        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=config.classical.dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Memory components
        self.memory = ClassicalMemoryInterface(
            config=config,
            memory_size=memory_size,
            content_dim=hidden_size
        )
        
        # Attention for memory access
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=config.classical.dropout,
            batch_first=True
        )
        
        # Memory integration layers
        self.memory_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.classical.dropout)
        
    def initialize(self) -> None:
        """Initialize the classical LSTM and memory components."""
        super().initialize()
        self.memory.initialize()
        
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Initialize forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass through memory-augmented LSTM.
        
        Args:
            input_sequence: Input tensor (batch_size, seq_len, input_size)
            hidden_state: Optional initial hidden state
            use_memory: Whether to use memory augmentation
            
        Returns:
            Tuple of (output, hidden_state, memory_info)
        """
        batch_size, seq_len, _ = input_sequence.shape
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(input_sequence, hidden_state)
        
        memory_info = {
            'memory_used': False,
            'memory_hits': 0,
            'memory_writes': 0,
            'attention_weights': []
        }
        
        if use_memory:
            # Memory-augmented processing
            enhanced_output, memory_info = self._memory_augmented_forward(
                lstm_output, input_sequence
            )
            
            # Combine LSTM and memory outputs
            combined_output = self._combine_outputs(lstm_output, enhanced_output)
        else:
            combined_output = lstm_output
        
        # Apply layer normalization and dropout
        output = self.layer_norm(combined_output)
        output = self.dropout(output)
        
        return output, hidden_state, memory_info
    
    def _memory_augmented_forward(
        self,
        lstm_output: torch.Tensor,
        input_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform memory-augmented processing."""
        batch_size, seq_len, hidden_size = lstm_output.shape
        
        enhanced_outputs = []
        total_hits = 0
        total_writes = 0
        attention_weights_list = []
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                current_state = lstm_output[batch_idx, seq_idx]
                
                # Query memory
                retrieved_memory, similarities = self.memory.read(
                    query=current_state,
                    k=5  # Retrieve top 5 matches
                )
                
                if len(retrieved_memory) > 0:
                    # Apply attention to retrieved memories
                    memory_tensor = retrieved_memory.unsqueeze(0)  # Add batch dim
                    query_tensor = current_state.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
                    
                    attended_memory, attention_weights = self.memory_attention(
                        query_tensor, memory_tensor, memory_tensor
                    )
                    
                    enhanced_state = attended_memory.squeeze(0).squeeze(0)
                    attention_weights_list.append(attention_weights.detach())
                    total_hits += 1
                else:
                    enhanced_state = torch.zeros_like(current_state)
                
                enhanced_outputs.append(enhanced_state)
                
                # Write current state to memory
                self.memory.write(current_state)
                total_writes += 1
        
        # Stack enhanced outputs
        enhanced_tensor = torch.stack(enhanced_outputs).view(batch_size, seq_len, hidden_size)
        
        memory_info = {
            'memory_used': True,
            'memory_hits': total_hits,
            'memory_writes': total_writes,
            'attention_weights': attention_weights_list,
            'memory_utilization': self.memory.get_memory_utilization()
        }
        
        return enhanced_tensor, memory_info
    
    def _combine_outputs(
        self,
        lstm_output: torch.Tensor,
        memory_output: torch.Tensor
    ) -> torch.Tensor:
        """Combine LSTM and memory outputs using gating mechanism."""
        # Concatenate outputs
        combined = torch.cat([lstm_output, memory_output], dim=-1)
        
        # Apply gating
        gate = torch.sigmoid(self.memory_gate(combined))
        gated_memory = gate * memory_output
        
        # Final combination
        final_combined = torch.cat([lstm_output, gated_memory], dim=-1)
        output = self.output_projection(final_combined)
        
        return output
    
    def clear_memory(self) -> None:
        """Clear the classical memory."""
        self.memory.clear()
        self.logger.debug("Classical memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory.get_memory_stats()


class MemoryAugmentedLSTMCell(nn.Module):
    """
    Single LSTM cell with memory augmentation.
    
    Provides fine-grained control over memory operations
    at the cell level.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int = 512
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Standard LSTM gates
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Memory components
        self.memory_matrix = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.memory_attention = nn.Linear(hidden_size, memory_size)
        self.memory_update = nn.Linear(hidden_size, hidden_size)
        
        # Initialize memory
        nn.init.xavier_uniform_(self.memory_matrix)
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through memory-augmented LSTM cell.
        
        Args:
            input_tensor: Input tensor (batch_size, input_size)
            hidden_state: Tuple of (hidden, cell) states
            
        Returns:
            Tuple of (new_hidden, new_cell, attention_weights)
        """
        hidden, cell = hidden_state
        batch_size = input_tensor.size(0)
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, hidden], dim=1)
        
        # Compute gates
        i_gate = torch.sigmoid(self.input_gate(combined))
        f_gate = torch.sigmoid(self.forget_gate(combined))
        c_gate = torch.tanh(self.cell_gate(combined))
        o_gate = torch.sigmoid(self.output_gate(combined))
        
        # Memory attention
        attention_scores = self.memory_attention(hidden)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Read from memory
        memory_read = torch.matmul(attention_weights, self.memory_matrix)
        
        # Update cell state with memory
        new_cell = f_gate * cell + i_gate * c_gate + 0.1 * memory_read
        
        # Compute new hidden state
        new_hidden = o_gate * torch.tanh(new_cell)
        
        # Update memory (simplified)
        memory_update = self.memory_update(new_hidden)
        self.memory_matrix.data += 0.01 * torch.matmul(
            attention_weights.t(), 
            memory_update.unsqueeze(1).expand(-1, self.memory_size, -1).mean(dim=0)
        )
        
        return new_hidden, new_cell, attention_weights


class BidirectionalMemoryLSTM(ClassicalComponent):
    """
    Bidirectional LSTM with memory augmentation.
    
    Processes sequences in both forward and backward directions
    with shared memory access.
    """
    
    def __init__(
        self,
        config,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        memory_size: int = 1024,
        name: str = "BidirectionalMemoryLSTM"
    ):
        super().__init__(config, name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_size = memory_size
        
        # Forward and backward LSTMs
        self.forward_lstm = ClassicalLSTM(
            config=config,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            memory_size=memory_size // 2,
            name=f"{name}_Forward"
        )
        
        self.backward_lstm = ClassicalLSTM(
            config=config,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            memory_size=memory_size // 2,
            name=f"{name}_Backward"
        )
        
        # Output combination
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def initialize(self) -> None:
        """Initialize bidirectional LSTM components."""
        super().initialize()
        self.forward_lstm.initialize()
        self.backward_lstm.initialize()
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through bidirectional memory LSTM.
        
        Args:
            input_sequence: Input tensor (batch_size, seq_len, input_size)
            use_memory: Whether to use memory augmentation
            
        Returns:
            Tuple of (output, combined_memory_info)
        """
        # Forward pass
        forward_output, _, forward_info = self.forward_lstm(
            input_sequence, use_memory=use_memory
        )
        
        # Backward pass (reverse sequence)
        reversed_sequence = torch.flip(input_sequence, dims=[1])
        backward_output, _, backward_info = self.backward_lstm(
            reversed_sequence, use_memory=use_memory
        )
        
        # Reverse backward output to align with forward
        backward_output = torch.flip(backward_output, dims=[1])
        
        # Combine outputs
        combined_output = torch.cat([forward_output, backward_output], dim=-1)
        output = self.output_projection(combined_output)
        
        # Combine memory info
        combined_info = {
            'forward_memory': forward_info,
            'backward_memory': backward_info,
            'total_memory_hits': forward_info.get('memory_hits', 0) + backward_info.get('memory_hits', 0),
            'total_memory_writes': forward_info.get('memory_writes', 0) + backward_info.get('memory_writes', 0)
        }
        
        return output, combined_info
    
    def clear_memory(self) -> None:
        """Clear memory for both forward and backward LSTMs."""
        self.forward_lstm.clear_memory()
        self.backward_lstm.clear_memory()
        self.logger.debug("Bidirectional memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        return {
            'forward_stats': self.forward_lstm.get_memory_stats(),
            'backward_stats': self.backward_lstm.get_memory_stats()
        }
