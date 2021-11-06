# Example Content

## Example Math

$$
\sum_{i=1}^n x_i^2
$$

## Example Admonitions

!!! warning "Be Very Careful"

    You should be very careful not to use too many admonitions ;)

## Example Table

| City      | Country        | Population   |
| --        | --             | --           |
| London    | United Kingdom | 8.89 million |
| New York  | United States  | 8.42 million |
| Singapore | Singapore      | 5.69 million |

## Example Code Sample

Code samples can contain line numbers and highlight important lines.

```python linenums="1" hl_lines="6 7"
def factorial(x: int) -> int:
    """Compute the factorial of a non-negative integer."""
    if x < 0:
        raise ValueError("x must be non-negative")
    
    acc = 1
    while x > 0:
        acc = x * acc
        x -= 1
    
    return acc
```
