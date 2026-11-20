import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def recode_variable( df, codings_df, data_dict_df, column_name, replace_name: str = None, convert_to_dummies: bool = True ):
    """
    Recode a variable in the dataframe based on the codings and data dictionary.
    If replace_name is provided, the recoded column will replace the original column.
    Otherwise, a new column with suffix '_recoded' will be created.
    """
    coding_id = data_dict_df.loc[ data_dict_df[ 'name' ] == column_name, 'coding_name' ].values[ 0 ]
    variable_type = data_dict_df.loc[ data_dict_df[ 'name' ] == column_name, 'type' ].values[ 0 ]
    coding_map = codings_df[ codings_df[ 'coding_name' ] == coding_id ][ [ 'code', 'meaning' ] ].set_index( 'code' )[ 'meaning' ].to_dict()
    is_multi_select = data_dict_df.loc[ data_dict_df[ 'name' ] == column_name, 'is_multi_select' ].values[ 0 ] == 'yes'

    print( f"Recoding variable '{column_name}' using coding '{coding_id}' of type '{variable_type}'." )

    if variable_type == 'integer' and not is_multi_select:
        # convert keys to integers if they are not already in the map
        coding_map = { int( k ): v for k, v in coding_map.items() }

    if replace_name is None:
        new_column_name = column_name
    else:
        new_column_name = replace_name

    if not is_multi_select:
        df[ new_column_name ] = df[ column_name ].map( coding_map )

        if convert_to_dummies:
            df = categorical_to_dummies( df, new_column_name )

    elif convert_to_dummies:
        # value of the column is a list
        # split the list and map the values
        dummies = df[ column_name ].str.replace( '[', '', regex = False ).str.replace( ']', '', regex = False ).str.replace( " ", "", regex = False ).str.split( ',' ) \
                .explode().dropna().map( coding_map ).str.replace( r'\s+', '_', regex = True ).str.replace(
                '-', '_', regex = False ).str.strip().pipe( pd.get_dummies ).groupby( level = 0 ).max()
        # set column names to have a prefix of the original column name
        dummies.columns = [ f"{new_column_name}__{col}" for col in dummies.columns ]
        # convert to int
        dummies = dummies.astype( int )
        # join back to the original dataframe
        df = df.join( dummies )

    return df


def categorical_to_dummies( df, column_name ):
    """
    Convert a categorical column to dummy/indicator variables.
    """
    # Convert values in the column, by removing spaces and special characters
    df[ column_name ] = df[ column_name ].str.replace( r'\s+', '_', regex = True ).str.replace( '-', '_', regex = False ).str.strip()
    dummies = pd.get_dummies( df[ column_name ], prefix = column_name, prefix_sep = '__', dummy_na = False, dtype = int )
    # Where in the original column the value was NaN, set all dummies to NaN
    dummies.loc[ df[ column_name ].isna(), : ] = np.nan
    df = pd.concat( [ df, dummies ], axis = 1 )
    df.drop( columns = [ column_name ], inplace = True )
    return df


def columns_with_prefix( df, prefix, omit_prefer_not: bool = True ):
    return [ col for col in df.columns if col.startswith( prefix ) and ( not omit_prefer_not or 'Prefer_not' not in col ) ]


class ResidualZScaler:
    """
    Residualize multiple target columns w.r.t. covariates,
    then z-score the residuals (fit on training data only).
    """

    def __init__( self, target_cols, covariates, min_n = 50, output_suffix = "_resid_z" ):
        self.target_cols = list( target_cols )
        self.covariates = list( covariates )
        self.min_n = min_n
        self.output_suffix = output_suffix

        self.models_ = {} # OLS models per variable
        self.means_ = {}  # mean of residuals (train)
        self.stds_ = {}   # std of residuals (train)

    def fit( self, df ):
        for col in self.target_cols:
            cols = [ col ] + self.covariates
            fit_df = df[ cols ].dropna()

            if len( fit_df ) < self.min_n:
                self.models_[ col ] = None
                self.means_[ col ] = np.nan
                self.stds_[ col ] = np.nan
                continue

            # 1) fit residualization model
            formula = f"{col} ~ " + " + ".join( self.covariates )
            model = smf.ols( formula, data = fit_df ).fit()
            self.models_[ col ] = model

            # 2) compute residuals on *training data*
            pred = model.predict( fit_df[ self.covariates ] )
            resid = fit_df[ col ] - pred

            # 3) estimate scaling parameters
            mu = resid.mean()
            sd = resid.std( ddof = 0 )

            # guard against degenerate cases
            if not np.isfinite( sd ) or sd == 0:
                sd = np.nan

            self.means_[ col ] = mu
            self.stds_[ col ] = sd

        return self

    def transform( self, df ):
        out = pd.DataFrame( index = df.index )

        for col in self.target_cols:
            model = self.models_.get( col )
            mu = self.means_.get( col )
            sd = self.stds_.get( col )

            # default: all NaN
            z = pd.Series( np.nan, index = df.index, dtype = float )

            out_name = f"{col}{self.output_suffix}"

            if model is None or not np.isfinite( sd ):
                out[ out_name ] = z
                continue

            # predict expected score where covariates are present
            cov_ok = df[ self.covariates ].notna().all( axis = 1 )
            pred = pd.Series( np.nan, index = df.index, dtype = float )
            pred.loc[ cov_ok ] = model.predict( df.loc[ cov_ok, self.covariates ] )

            # residuals
            resid = df[ col ] - pred
            resid[ df[ col ].isna() ] = np.nan

            # z-score
            z = ( resid - mu ) / sd

            out[ out_name ] = z

        return out

    def fit_transform( self, df ):
        return self.fit( df ).transform( df )


### Usage example:
# CF_cols = [c for c in CF_at_assessment if c != "eid"]
# covariates = ["age_at_assessment", "sex"]

# train_df = merged.loc[train_idx]
# valid_df = merged.loc[valid_idx]

# rz = ResidualZScaler(
#     target_cols=CF_cols,
#     covariates=covariates,
#     min_n=100
# )

# rz.fit(train_df)

# train_z = rz.transform(train_df)
# valid_z = rz.transform(valid_df)
