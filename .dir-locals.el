;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((python-mode . ((eval . (pythonic-activate "/home/ethan/miniconda3/envs/torch1.3"))
                 (flycheck-checker . lsp)
                 (flycheck-python-flake8-executable . "/home/ethan/miniconda3/envs/tf/bin/flake8")
                 (flycheck-flake8rc . "/home/ethan/code/BiSeNet/.flake8")
                 (my/pydocstylerc . "/home/ethan/code/BiSeNet/.pydocstylerc")
                 (flycheck-disabled-checkers . python-mypy)
                 (eval . (progn
                               (flycheck-add-next-checker 'lsp 'python-flake8)
                         (setq flycheck-disabled-checkers
                                 '(python-mypy python-pylint python-pycompile))
                               ))
                 (format-all-formatters . (("Python"
                                                     (yapf "--style" "/home/ethan/code/BiSeNet/.style.yapf"))))
(c-mode-common . (lambda () (flycheck-add-next-checker 'lsp 'c/c++-clang-tidy
                                 (python-indent-offset . 2)))))
