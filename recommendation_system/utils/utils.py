def normalize(
        value: float,
        old_max: float,
        old_min: float,
        new_max:float = 10.0,
        new_min: float = 0.0
) -> float:
  old_range = (old_max - old_min)
  new_range = (new_max - new_min)
  return (((value - old_min) * new_range) / old_range) + new_min