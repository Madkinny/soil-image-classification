# Data audit

| Class | Files found | Used after cleaning |
|---|---|---|
| Alluvial soil | 54 | 49 |
| Black Soil | 116 | 91 |
| Clay soil | 65 | 53 |
| **Total** | **235** | **193** |

- Exact duplicates removed: **40**
- Near-duplicate groups kept together in one split: **8**
- Images dropped because the same photo had different labels: **2**
- Split sizes: {'train': 138, 'val': 28, 'test': 27}
- Duplicate groups shared between splits: **0**

## Label conflicts

- `Alluvial soil/image16 (2).jpeg` (Alluvial soil); `Clay soil/Copy of clay-soil-close-up-brown-179039817.jpg` (Clay soil)
