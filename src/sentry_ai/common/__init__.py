"""Cross-cutting utilities shared by every layer: logging and exceptions.

This package must never import from ``domain``, ``simulation``, or any AI
package — see PROJECT.md §2 "Architectural Style" for the dependency rule.
"""
