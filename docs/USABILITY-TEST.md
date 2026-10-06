# Usability test: from zero to a first number

The ease-of-use work (phases 0 to 2) was designed at a desk. This is the check that it
works on real people. Two or three sessions are enough to find the top three problems;
fix those, then run two more sessions.

## Who

People who match the audience the work is for, not colleagues who already know the
project:

- **Reader:** someone who wants to know "is 4-bit okay?" and will not install anything.
- **Operator:** someone who can paste commands into a terminal and follow instructions,
  but has never used Python tooling.
- **Non-terminal user:** someone who would close a terminal window if one opened.

One person per type is a good first round.

## Setup

- A machine they have not used for this before. For the operator, a fresh clone in a new
  folder. For the non-terminal user, `./install.sh` already run and the desktop shortcut
  installed (`scripts/desktop-shortcut.sh`), or a laptop with the browser page already open.
- The tester sits beside them and does not help unless they are stuck for more than three
  minutes. Write down what they say and where they look.
- Start a stopwatch when they first see the README (reader, operator) or the page
  (non-terminal user).

## Tasks

| who | task | done when |
|---|---|---|
| reader | "Is opt-125m still usable at 4 bits? Which method is best?" using only the website or `results/` | they name a method and a percentage, and say what the percentage means |
| operator | "Get one measurement of your own on this machine" starting from the README | a result row exists and they can say what the verdict sentence means |
| non-terminal user | "Find out how much this model loses at 4 bits" starting from the application menu or the page | they read the verdict sentence back in their own words |

## What to record

For every session, one row:

| field | what to write |
|---|---|
| time to first number | stopwatch reading when the task's "done when" is met |
| questions asked | each question, verbatim, and whether the answer was on screen at the time |
| stuck points | every pause over one minute: what was on screen, what they tried |
| wrong turns | files or menus they opened that were not the right ones |
| words they did not know | any term they asked about or skipped past |
| their own words for the result | how they described the result afterwards |

Keep the rows in `docs/usability/<date>-<who>.md` (the folder is yours to create).

## Deciding what to fix

1. Pool the stuck points and questions from all sessions.
2. Rank by how many people hit them, then by time lost.
3. Fix the top three. Nothing else, until the next round.
4. Run two more sessions on the fixed version and compare times.

Targets, so the rounds have a goal: a reader finds the method and percentage in under
three minutes; an operator reaches a verdict sentence in under fifteen minutes including
the install; a non-terminal user reaches it in under five minutes with the page open.

## Guarding the words

`tests/test_copy.py` keeps the plain-language surfaces honest between rounds: no raw
keys in menus, every failure reason ends with a fix, every link in the guides resolves.
When a session shows a word people do not know, add it to the jargon list there so it
cannot creep back in.
