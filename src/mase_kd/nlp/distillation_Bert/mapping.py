def generate_layer_mapping(num_student_layers, num_teacher_layers):
    """
    Automatically generates a layer mapping dictionary between student and teacher.
    Uses the uniform mapping strategy recommended by the TinyBERT paper.
    
    Example: 
        student=4, teacher=12 
        Returns: {0: 2, 1: 5, 2: 8, 3: 11}
        
    Args:
        num_student_layers (int): Number of hidden layers in the student model.
        num_teacher_layers (int): Number of hidden layers in the teacher model.
        
    Returns:
        dict: A mapping from student layer index to teacher layer index.
    """
    if num_teacher_layers % num_student_layers != 0:
        raise ValueError("For uniform mapping, the number of teacher layers must be a multiple of the student layers!")
        
    interval = num_teacher_layers // num_student_layers
    mapping = {}
    
    for i in range(num_student_layers):
        # Mapping formula: (student_layer_index + 1) * interval - 1
        teacher_idx = (i + 1) * interval - 1
        mapping[i] = teacher_idx
        
    return mapping