Nova font
=========


Here is the Nova font designed by `Wojciech Kalinowski
<http://luc.devroye.org/fonts-57186.html>`_ and downloaded from the
`Font Library <https://fontlibrary.org/en/font/nova>`_.
It has nice symmetric glyphs, such as "p" and "d", and "m" and "w".


Custom font with LaTeX
-----------------------

* https://www.overleaf.com/latex/examples/example-custom-font/htswqdkhqxjk
* https://tex.stackexchange.com/a/338067
* https://latexref.xyz/Low_002dlevel-font-commands.html
* https://www.latex-project.org/help/documentation/fntguide.pdf
* https://www.overleaf.com/learn/latex/Questions/I_have_a_custom_font_I%27d_like_to_load_to_my_document._How_can_I_do_this%3F

.. code:: sh

  ttf2tfm nova.ttf -p T1-WGL4.enc

``T1-WGL4.enc`` is distributed with TeX Live.


pdfTeX error: unknown version of OS/2 table (0004)
--------------------------------------------------

Designed with FontForge, Nova ttf files trigger the above errors of
pdfTeX, as described in https://tex.stackexchange.com/a/225089.
These files have been "fixed" with FontForge following Reuben Thomas'
comment to that answer.
