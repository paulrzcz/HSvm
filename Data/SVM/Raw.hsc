{-# LANGUAGE ForeignFunctionInterface, GeneralizedNewtypeDeriving, 
             EmptyDataDecls #-}

#include "svm.h"
#include <stddef.h>
#let alignment t = "%lu", (unsigned long)offsetof(struct {char x__; t (y__); }, y__)

module Data.SVM.Raw where

-- TODO limitare l'export
-- TODO verificare l'import

import Foreign.Storable (Storable(..), peekByteOff, pokeByteOff)
import Foreign.C.Types (CDouble, CInt)
import Foreign.C.String (CString)
import Foreign.Ptr(nullPtr, Ptr)
import Foreign.ForeignPtr (FinalizerPtr)

data CSvmNode = CSvmNode { 
    index:: CInt,
    value:: CDouble 
}

instance Storable CSvmNode where
    sizeOf _ = #size struct svm_node
    alignment _ = #alignment struct svm_node
    peek ptr = do index <- (#peek struct svm_node, index) ptr
                  value <- (#peek struct svm_node, value) ptr
                  return $ CSvmNode index value
    poke ptr (CSvmNode i v) = do (#poke struct svm_node, index) ptr i
                                 (#poke struct svm_node, value) ptr v

data CSvmProblem = CSvmProblem {
    l:: CInt,
    y:: Ptr CDouble,
    x:: Ptr (Ptr CSvmNode)
}       

instance Storable CSvmProblem where
    sizeOf _ = #size struct svm_problem
    alignment _ = #alignment struct svm_problem
    peek ptr = do l <- (#peek struct svm_problem, l) ptr
                  y <- (#peek struct svm_problem, y) ptr
                  x <- (#peek struct svm_problem, x) ptr
                  return $ CSvmProblem l y x
    poke ptr (CSvmProblem l y x) = do (#poke struct svm_problem, l) ptr l
                                      (#poke struct svm_problem, y) ptr y
                                      (#poke struct svm_problem, x) ptr x


-- TODO esportare solo il tipo e non il costruttore?
newtype CSvmType = CSvmType {unCSvmType :: CInt}
                   deriving (Storable, Show)
#enum CSvmType, CSvmType, C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR

newtype CKernelType = CKernelType {unCKernelType :: CInt} 
                      deriving (Storable, Show)
#enum CKernelType, CKernelType, LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED

data CSvmParameter = CSvmParameter {
    svm_type     :: CSvmType,
    kernel_type  :: CKernelType,
    degree       :: CInt,
    gamma        :: CDouble,
    coef0        :: CDouble,
    cache_size   :: CDouble,
    eps          :: CDouble,
    c            :: CDouble,
    nr_weight    :: CInt,
    weight_label :: Ptr CInt,
    weight       :: Ptr CDouble,
    nu           :: CDouble,
    p            :: CDouble,
    shrinking    :: CInt,
    probability  :: CInt
} deriving Show

defaultCParam = CSvmParameter cSvc rbf 3 0 0 100 1e-3 1 
                              0 nullPtr nullPtr 0.5 0.1 1 0

instance Storable CSvmParameter where
    sizeOf _ = #size struct svm_parameter
    alignment _ = #alignment struct svm_parameter
    peek ptr = do svm_type     <- (#peek struct svm_parameter, svm_type) ptr
                  kernel_type  <- (#peek struct svm_parameter, kernel_type) ptr
                  degree       <- (#peek struct svm_parameter, degree) ptr
                  gamma        <- (#peek struct svm_parameter, gamma) ptr
                  coef0        <- (#peek struct svm_parameter, coef0) ptr
                  cache_size   <- (#peek struct svm_parameter, cache_size) ptr
                  eps          <- (#peek struct svm_parameter, eps) ptr
                  c            <- (#peek struct svm_parameter, C) ptr
                  nr_weight    <- (#peek struct svm_parameter, nr_weight) ptr
                  weight_label <- (#peek struct svm_parameter, weight_label) ptr
                  weight       <- (#peek struct svm_parameter, weight) ptr
                  nu           <- (#peek struct svm_parameter, nu) ptr
                  p            <- (#peek struct svm_parameter, p) ptr
                  shrinking    <- (#peek struct svm_parameter, degree) ptr
                  probability  <- (#peek struct svm_parameter, probability) ptr
                  return $ CSvmParameter svm_type kernel_type degree      
                                gamma coef0 cache_size eps c nr_weight
                                weight_label weight nu p shrinking probability
    poke ptr (CSvmParameter svm_type kernel_type degree
                           gamma coef0 cache_size eps c nr_weight
                           weight_label weight nu p shrinking probability) =
           do (#poke struct svm_parameter, svm_type) ptr svm_type
              (#poke struct svm_parameter, kernel_type) ptr kernel_type
              (#poke struct svm_parameter, degree) ptr degree
              (#poke struct svm_parameter, gamma) ptr gamma
              (#poke struct svm_parameter, coef0) ptr coef0
              (#poke struct svm_parameter, cache_size) ptr cache_size
              (#poke struct svm_parameter, eps) ptr eps
              (#poke struct svm_parameter, C) ptr c
              (#poke struct svm_parameter, nr_weight) ptr nr_weight
              (#poke struct svm_parameter, weight_label) ptr weight_label
              (#poke struct svm_parameter, weight) ptr weight
              (#poke struct svm_parameter, nu) ptr nu
              (#poke struct svm_parameter, p) ptr p
              (#poke struct svm_parameter, shrinking) ptr shrinking
              (#poke struct svm_parameter, probability) ptr probability

data CSvmModel

-- TODO cambiare il return type da 
foreign import ccall unsafe "svm.h svm_train" c_svm_train :: Ptr CSvmProblem -> Ptr CSvmParameter -> IO (Ptr CSvmModel)
                        
foreign import ccall unsafe "svm.h svm_cross_validation" c_svm_cross_validation:: Ptr CSvmProblem -> Ptr CSvmParameter -> CInt -> Ptr CDouble -> IO () 

foreign import ccall unsafe "svm.h svm_predict" c_svm_predict :: Ptr CSvmModel -> Ptr CSvmNode -> CDouble

foreign import ccall unsafe "svm.h svm_save_model" c_svm_save_model :: CString -> Ptr CSvmModel -> IO CInt

foreign import ccall unsafe "svm.h svm_load_model" c_svm_load_model :: CString -> IO (Ptr CSvmModel)
                        
foreign import ccall unsafe "svm.h svm_check_parameter" c_svm_check_parameter :: Ptr CSvmProblem -> Ptr CSvmParameter -> CString

foreign import ccall unsafe "svm.h &svm_destroy_model" c_svm_destroy_model :: FinalizerPtr CSvmModel

foreign import ccall unsafe "svm.h clone_model_support_vectors" c_clone_model_support_vectors :: Ptr CSvmModel -> IO ()
