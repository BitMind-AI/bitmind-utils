# Add this to synthetic_data_generator.py

    def generate_from_prompt(
        self,
        prompt: str,
        task: Optional[str] = None,
        image: Optional[Image.Image] = None,
        generate_at_target_size: bool = False
    ) -> Dict[str, Any]:
        """Generate synthetic data based on a provided prompt.
        
        Args:
            prompt: The text prompt to use for generation
            task: Optional task type ('t2i', 't2v', 'i2i')
            image: Optional input image for i2i generation
            generate_at_target_size: If True, generate at TARGET_IMAGE_SIZE dimensions
            
        Returns:
            Dictionary containing generated data information
        """
        bt.logging.info(f"Generating synthetic data from provided prompt: {prompt}")
        
        # Default to t2i if task is not specified
        if task is None:
            task = 't2i'
        
        # If model_name is not specified, select one based on the task
        if self.model_name is None and self.use_random_model:
            bt.logging.warning(f"No model configured. Using random model.")
            if task == 't2i':
                model_candidates = T2I_MODEL_NAMES
            elif task == 't2v':
                model_candidates = T2V_MODEL_NAMES
            elif task == 'i2i':
                model_candidates = I2I_MODEL_NAMES
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            self.model_name = random.choice(model_candidates)
        
        # Run the generation with the provided prompt
        gen_data = self._run_generation(
            prompt=prompt, 
            task=task, 
            model_name=self.model_name,
            image=image,
            generate_at_target_size=generate_at_target_size
        )
        
        # Clean up GPU memory
        self.clear_gpu()
        
        return gen_data